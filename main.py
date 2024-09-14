import image_recognition
import ingredients_synonims
import database_and_finalization

image_recognition.unpload_photo_to_recognize()
image_recognition.image_preprocessing()
image_recognition.recognition()
image_recognition.verification()

ingredients_synonims.synonim_search()

database_and_finalization.searching_for_ingredients()