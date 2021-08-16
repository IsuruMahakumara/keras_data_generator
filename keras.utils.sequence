class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels = None, batch_size= 32, dim=(1638,256), n_channels=1,
                 n_classes=2, shuffle=True, train_gen = True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.train_gen = train_gen
        self.on_epoch_end()
        
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        if self.train_gen:
            X, y = self.__data_generation(list_IDs_temp)
            return X, y
        else:
            return X
        

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def get_add(self,ID):
        drt = ID[0]
        if self.train_gen:
            add = "../input/seti-breakthrough-listen/train/"+drt+'/'+ID+'.npy'
        else :
            add = "../input/seti-breakthrough-listen/test/"+drt+'/'+ID+'.npy'
            
        return add

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
       
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(get_add(ID)).flatten().reshape(1638,256,1)

            # Store class
            y[i] = self.labels[i,1]
        return X, y
