import image_face

# image_face.image_client("TAI", "images/*.png", "output")
# image_face.image_server("TAI")

image_face.beauty_predict("images/*.png", "output/beauty")
image_face.detect_predict("images/*.png", "output/detect")
