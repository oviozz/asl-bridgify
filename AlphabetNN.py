import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

data_dir = '/Users/aahilali/Desktop/ASL_Dataset 2'
data = []
labels = []

# Check if the directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"The directory '{data_dir}' does not exist. Please provide the correct path to your dataset.")

# Function to process a directory and extract landmarks
def process_directory(directory):
    for root, _, files in os.walk(directory):
        for img_file in files:
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                data_aux = []
                img_path = os.path.join(root, img_file)
                img = cv2.imread(img_path)

                # Check if the image was read successfully
                if img is None:
                    print(f"Warning: Unable to read image '{img_path}'. Skipping.")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for landmark in hand_landmarks.landmark:
                            data_aux.append(landmark.x)
                            data_aux.append(landmark.y)
                    data.append(data_aux)
                    labels.append(os.path.basename(root))

# Process the train and test directories
train_dir = os.path.join(data_dir, 'Train')
test_dir = os.path.join(data_dir, 'Test')

if os.path.exists(train_dir):
    process_directory(train_dir)
else:
    print(f"Warning: The directory '{train_dir}' does not exist.")

if os.path.exists(test_dir):
    process_directory(test_dir)
else:
    print(f"Warning: The directory '{test_dir}' does not exist.")

# Check if all data samples have the same length
max_length = max(len(sample) for sample in data)
print(f"Max length of data samples: {max_length}")

# Pad or truncate data samples to the same length
data_padded = np.array([np.pad(sample, (0, max_length - len(sample))) if len(sample) < max_length else np.array(sample[:max_length]) for sample in data])
labels = np.array(labels)

print(f"Shape of data_padded: {data_padded.shape}")
print(f"Shape of labels: {labels.shape}")

# Save data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data processing complete and saved to 'data.pickle'")

# data_dir = '/Users/aahilali/Desktop/ASL_Dataset'
for i in sorted(os.listdir(data_dir)):
    if i == '.DS_Store':
        continue
    for j in os.listdir(os.path.join(data_dir, i))[0:1]:
        img_path = os.path.join(data_dir, i, j)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_rgb,  # img to draw
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        plt.figure()
        plt.title(i)
        plt.imshow(img_rgb)
        
plt.show()

# Split data
print('################got here#################')
X_train, X_test, y_train, y_test = train_test_split(np.array(data), labels, test_size=0.15, random_state=22, shuffle=True)
print('got here')

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize the model
rf = RandomForestClassifier(random_state=22)

# Perform grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_rf = grid_search.best_estimator_

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Fit the model with the best parameters
best_rf.fit(X_train, y_train)
print('got here2')

# Predict
pred = best_rf.predict(X_test)
print('got here 3')

# Accuracy
accuracy = accuracy_score(y_test, pred)
print(f'Accuracy: {accuracy}')
print('got here 4')

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': best_rf}, f)

print('got here 5')

# load model
model_dict = pickle.load(open('model.p','rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():

        data_aux=[]
        x_ = []
        y_ = []

        ret, frame = cap.read()
        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True 
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_rgb, # img to draw
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(28, 255, 3), thickness=5, circle_radius=10),
                    mp_drawing.DrawingSpec(color=(236, 255, 3), thickness=5, circle_radius=10)
                )


            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

            x1 = int(min(x_) * W)-10
            y1 = int(min(y_) * H)-10

            x2 = int(max(x_) * W)-10
            y2 = int(max(y_) * H)-10
            prediction = model.predict([np.array(data_aux)[0:42]])[0]

            cv2.rectangle(frame_rgb, (x1,y1-10), (x2,y2), (255,99,173), 6)
            cv2.putText(frame_rgb, prediction, (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 5, (255,0,0), 5, cv2.LINE_AA)

        cv2.imshow('frame',frame_rgb)  
        # cv2.waitKey(1)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()