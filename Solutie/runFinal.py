from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *


params: Parameters = Parameters()
params.dim_window = 64  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
params.dim_hog_cell = 8  # dimensiunea celulei
params.overlap = 0.3


# params.number_positive_examples = 1665 # barney
# params.number_positive_examples = 1001 # betty
# params.number_positive_examples = 1001 # fred
# params.number_positive_examples = 998 # wilma
params.number_positive_examples = 6974  # numarul exemplelor pozitive
params.number_negative_examples = 50000  # numarul exemplelor negative

params.threshold = 2.5 # toate ferestrele cu scorul > threshold si maxime locale devin detectii
params.has_annotations = True

params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite

if params.use_flip_images:
    params.number_positive_examples *= 2


personaje = ['all_faces', 'barney', 'betty', 'fred', 'wilma']

for personaj in personaje:
    facial_detector: FacialDetector = FacialDetector(params)

    facial_detector.train_classifier([], [], personaj)

    detections, scores, file_names = facial_detector.run()
    if "all_faces" in personaj:
        np.save(f'rezultate/task1/detections_{personaj}.npy', detections)
        np.save(f'rezultate/task1/file_names_{personaj}.npy', file_names)
        np.save(f'rezultate/task1/scores_{personaj}.npy', scores)
    else:
        np.save(f'rezultate/task2/detections_{personaj}.npy', detections)
        np.save(f'rezultate/task2/file_names_{personaj}.npy', file_names)
        np.save(f'rezultate/task2/scores_{personaj}.npy', scores)
    


# Salveaza detectiile, scorurile si imaginile

# np.save('detections_all_faces.npy', detections)
# np.save('file_names_all_faces.npy', file_names)
# np.save('scores_all_faces.npy', scores)

# np.save('detections_barney.npy', detections)
# np.save('file_names_barney.npy', file_names)
# np.save('scores_barney.npy', scores)

# np.save('detections_betty.npy', detections)
# np.save('file_names_betty.npy', file_names)
# np.save('scores_betty.npy', scores)

# np.save('detections_fred.npy', detections)
# np.save('file_names_fred.npy', file_names)
# np.save('scores_fred.npy', scores)

# np.save('detections_wilma.npy', detections)
# np.save('file_names_wilma.npy', file_names)
# np.save('scores_wilma.npy', scores)

