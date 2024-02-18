import os

class Parameters:
    def __init__(self):
        self.base_dir = '../'
        # self.dir_pos_examples = os.path.join(self.base_dir, 'pozitive_total')
        # self.dir_neg_examples = os.path.join(self.base_dir, 'negative_total')
        self.dir_pos_examples = os.path.join(self.base_dir, 'antrenare_hog/pozitive_total')
        self.dir_neg_examples = os.path.join(self.base_dir, 'antrenare_hog/negative_total')

        #TODO : DE INLOCUIT
        self.dir_test_examples = os.path.join(self.base_dir, 'testare')  # 


        self.path_annotations = os.path.join(self.base_dir, 'task1_gt_validare.txt')
        # self.path_annotations = os.path.join(self.base_dir, 'task2_barney_gt_validare.txt')
        # self.path_annotations = os.path.join(self.base_dir, 'task2_betty_gt_validare.txt')
        # self.path_annotations = os.path.join(self.base_dir, 'task2_fred_gt_validare.txt')
        # self.path_annotations = os.path.join(self.base_dir, 'task2_wilma_gt_validare.txt')


        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 64  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.dim_descriptor_cell = 64  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 6974  # numarul exemplelor pozitive
        self.number_negative_examples = 10000  # numarul exemplelor negative
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 0

        self.testare = 0
        self.nr_test = 200
        # cate exemple sa foloseasca pentru validare
        if self.testare:
            self.nr_test = 1
