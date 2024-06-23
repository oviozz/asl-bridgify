import sign_language_translator as slt

# Print available language codes
print(slt.TextLanguageCodes, slt.SignLanguageCodes)

# Initialize the model for English and American Sign Language (ASL)
model = slt.models.ConcatenativeSynthesis(
    text_language="english", sign_language="us-asl", sign_format="video"
)

# Example text to translate
text = "This is very good."
sign = model.translate(text)  # tokenize, map, download & concatenate
sign.show()
sign.save(f"{text}.mp4")

model.text_language = "english"  # slt.TextLanguageCodes.ENGLISH  # slt.languages.text.English()
sign_2 = model.translate("Five hours.")
sign_2.show()

# Load the sign-to-text model (pytorch) (COMING SOON!)
# translation_model = slt.get_model(slt.ModelCodes.Gesture)
embedding_model = slt.models.MediaPipeLandmarksModel()

# Load the video file
sign = slt.Video("/Users/aahilali/Desktop/raw_videos/_2FBDaOPYig_1-3-rgb_front.mp4")
embedding = embedding_model.embed(sign.iter_frames())

# Translate embeddings to text (Note: This feature might be COMING SOON)
# text = translation_model.translate(embedding)
# print(text)

sign.show()
slt.Landmarks(embedding, connections="mediapipe-world").show()
