import os

################ Configuration File #######################

KNOWN_FACES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/known_faces')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/results')
TEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/test_faces')
DB_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/db/faces.json')
TOLERANCE = 0.55

##############################################################