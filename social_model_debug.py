class SocialModel():

    def __init__(self, infer=False):
        '''
        args 
        '''
        # If sampling new trajectories, then infer mode
        if infer:
            # Sample one position at a time
            batch_size = 1
            aseq_length = 1
        # Store the arguments
        self.infer = infer
        # Store rnn size and grid_size
        self.rnn_size = 128
        self.grid_size = 8
        # Maximum number of peds
        self.maxNumPeds =30
        self.seq_length=5
        self.maxNumPeds=40
        self.embedding_size=64
        self.output_size = 5
        self.learning_rate=0.0001
        self.grad_clip=0.5
        '''
        model
        '''
        with tf.name_scope("LSTM_cell"):
        #with tf.name_scope("LSTM_cell",reuse=True):
            cell = rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=False)
        self.input_data = tf.placeholder(tf.float32, 
                                         [self.seq_length, self.maxNumPeds, 3], name="input_data")
        #self.input_data维度为(seq_length,maxNumPeds，3)
        
        self.target_data = tf.placeholder(tf.float32, 
                                          [self.seq_length, self.maxNumPeds, 3], name="target_data")
        #self.target_data维度为(seq_length,maxNumPeds，3)
        
        # Grid data would be a binary matrix which encodes whether a pedestrian is present in
        # a grid cell of other pedestrian
        self.grid_data = tf.placeholder(tf.float32,
                                        [self.seq_length, self.maxNumPeds,
                                         self.maxNumPeds, self.grid_size*self.grid_size], name="grid_data")
        self.lr = tf.Variable(self.learning_rate, trainable=False, name="learning_rate")
        
        
        # Define LSTM states for each pedestrian
        # maxNumPeds每一帧最大行人数目
        with tf.variable_scope("LSTM_states"):
        #with tf.variable_scope("LSTM_states",reuse=True):
            self.LSTM_states = tf.zeros([self.maxNumPeds, cell.state_size], name="LSTM_states")
            self.initial_states = tf.split(axis=0,
                                           num_or_size_splits=self.maxNumPeds, value=self.LSTM_states)
            #将LSTM_state变成一个list，有MaxNumPeds个元素，每个为256维（一个lstm有128*2个维度hidden units）
        
        # Define hidden output states for each pedestrian
        with tf.variable_scope("Hidden_states"):
        #with tf.variable_scope("Hidden_states",reuse=True):
            # self.output_states = tf.zeros([args.maxNumPeds, cell.output_size], name="hidden_states")
            self.output_states = tf.split(axis=0, num_or_size_splits=self.maxNumPeds, value=tf.zeros([self.maxNumPeds, cell.output_size]))
            #将LSTM_state变成一个list，有MaxNumPeds个元素，每个为output_size维度）
    
    
        # List of tensors each of shape args.maxNumPedsx3 corresponding to each frame in the sequence
        with tf.name_scope("frame_data_tensors"):
        #with tf.name_scope("frame_data_tensors",reuse=True):
            # frame_data = tf.split(0, args.seq_length, self.input_data, name="frame_data")
            frame_data = [tf.squeeze(input_, [0]) 
                          for input_ in tf.split(axis=0, num_or_size_splits=self.seq_length, value=self.input_data)]
            #组成一个list，list包含seq_length个元素，每个元素shape为（maxNumPeds，3）
       
    
        with tf.name_scope("frame_target_data_tensors"):
        #with tf.name_scope("frame_target_data_tensors",reuse=True):
            # frame_target_data = tf.split(0, args.seq_length, self.target_data, name="frame_target_data")
            self.frame_target_data = [tf.squeeze(target_, [0]) 
                                      for target_ in tf.split(axis=0, num_or_size_splits=self.seq_length, value=self.target_data)]
            #组成一个list，list包含seq_length个元素，每个元素shape为（maxNumPeds，3）            
            
          
        with tf.name_scope("grid_frame_data_tensors"):
        #with tf.name_scope("grid_frame_data_tensors",reuse=True):
            # This would contain a list of tensors each of shape MNP x MNP x (GS**2) encoding the mask
            # grid_frame_data = tf.split(0, args.seq_length, self.grid_data, name="grid_frame_data")
            grid_frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(axis=0, num_or_size_splits=self.seq_length, value=self.grid_data)]    
            #组成一个list，list包含seq_length个元素，每个元素shape为（self.maxNumPeds, self.maxNumPeds, self.grid_size*self.grid_size）
  

        # Cost的一些参数
        with tf.name_scope("Cost_related_stuff"):
        #with tf.name_scope("Cost_related_stuff",reuse=True):
            self.cost = tf.constant(0.0, name="cost")
            self.counter = tf.constant(0.0, name="counter")
            self.increment = tf.constant(1.0, name="increment")
            
            
        # Containers to store output distribution parameters
        with tf.name_scope("Distribution_parameters_stuff"):
            # self.initial_output = tf.zeros([args.maxNumPeds, self.output_size], name="distribution_parameters")
            self.initial_output = tf.split(axis=0, num_or_size_splits=self.maxNumPeds, value=tf.zeros([self.maxNumPeds, self.output_size]))            
            #组成一个list，list包含seq_length个元素，每个元素shape为（self.output_size）
            
        self.define_embedding_and_output_layers()
        #定义embedding的各种参数
        
        
        # Tensor to represent non-existent ped
        with tf.name_scope("Non_existent_ped_stuff"):
            nonexistent_ped = tf.constant(0.0, name="zero_ped")
        
        
        for seq, frame in enumerate(frame_data):
            print("Frame number", seq)
            current_frame_data = frame  
            # MNP x 3 tensor
            current_grid_frame_data = grid_frame_data[seq] 
            # MNP x MNP x (GS**2) tensor
            social_tensor = tf.zeros([
                    self.maxNumPeds, self.grid_size*self.grid_size*self.rnn_size])
            for ped in range(self.maxNumPeds):
                #print("Pedestrian Number", ped)
                pedID = current_frame_data[ped, 0]
                with tf.name_scope("extract_input_ped"):
                    self.spatial_input = tf.slice(current_frame_data, [ped, 1], [1, 2])
                    # Tensor of shape (1,2)
                    # Extract x and y positions of the current ped
                    self.tensor_input = tf.slice(social_tensor, 
                                                 [ped, 0],[1, self.grid_size*self.grid_size*self.rnn_size])
                    # current ped对应的social
                with tf.name_scope("embeddings_operations"):
                    embedded_spatial_input = tf.nn.relu(
                        tf.nn.xw_plus_b(self.spatial_input, self.embedding_w, self.embedding_b))
                    embedded_tensor_input = tf.nn.relu(
                        tf.nn.xw_plus_b(self.tensor_input, self.embedding_t_w, self.embedding_t_b))
                
                
                with tf.name_scope("concatenate_embeddings"): 
                    # Concatenate the embeddings
                    complete_input = tf.concat(axis=1, values=[embedded_spatial_input, embedded_tensor_input])
                

                with tf.variable_scope("LSTM") as scope:
                    if seq > 0 or ped > 0:
                        scope.reuse_variables()
                    self.output_states[ped], self.initial_states[ped] = cell(complete_input, self.initial_states[ped])
                    #第一个是cell的output_hidden，第二个是两个hidden_state拼起来
                    
                with tf.name_scope("output_linear_layer"):
                    self.initial_output[ped] = tf.nn.xw_plus_b(
                        self.output_states[ped], self.output_w, self.output_b)
                
                
                with tf.name_scope("extract_target_ped"):
                    # Extract x and y coordinates of the target data
                    # x_data and y_data would be tensors of shape 1 x 1
                    [x_data, y_data] = tf.split(axis=1, num_or_size_splits=2,value=tf.slice(self.frame_target_data[seq], [ped, 1], [1, 2]))
                                    
                
                
                with tf.name_scope("get_coef"):
                    [o_mux, o_muy, o_sx, o_sy, o_corr] = self.get_coef(self.initial_output[ped])
                
                with tf.name_scope("calculate_loss"):
                    lossfunc = self.get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)
 
                with tf.name_scope("increment_cost"):
                # If it is a non-existent ped, it should not contribute to cost
                    self.cost = tf.where(tf.equal(pedID, nonexistent_ped), 
                                  self.cost, tf.add(self.cost, lossfunc))
                    self.counter = tf.where(tf.not_equal(pedID, nonexistent_ped), 
                                     tf.add(self.counter, self.increment), self.counter) 
                    
        with tf.name_scope("mean_cost"):
            # Mean of the cost
            self.cost = tf.div(self.cost, self.counter)
            
        tvars = tf.trainable_variables()
        #把每个ped的state拼接起来
        self.final_states = tf.concat(axis=0, values=self.initial_states)
        
        # Get the final distribution parameters
        self.final_output = self.initial_output
        
        # Compute gradients
        self.gradients = tf.gradients(self.cost, tvars)
        
        # Clip the gradients
        grads, _ = tf.clip_by_global_norm(self.gradients, self.grad_clip)
        
        optimizer = tf.train.RMSPropOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        
    """
    Embedding
    """   
    def define_embedding_and_output_layers(self):
        # Define variables for the spatial coordinates embedding layer
        with tf.variable_scope("coordinate_embedding"):
            self.embedding_w = tf.get_variable("embedding_w", [2, self.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.embedding_b = tf.get_variable("embedding_b", [self.embedding_size], initializer=tf.constant_initializer(0.01))

        # Define variables for the social tensor embedding layer
        with tf.variable_scope("tensor_embedding"):
            self.embedding_t_w = tf.get_variable("embedding_t_w", [self.grid_size*self.grid_size*self.rnn_size, self.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.embedding_t_b = tf.get_variable("embedding_t_b", [self.embedding_size], initializer=tf.constant_initializer(0.01))

        # Define variables for the output linear layer
        with tf.variable_scope("output_layer"):
            self.output_w = tf.get_variable("output_w", [self.rnn_size, self.output_size], initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.output_b = tf.get_variable("output_b", [self.output_size], initializer=tf.constant_initializer(0.01))
            
    def getSocialTensor(self, grid_frame_data, output_states):
        '''
        Computes the social tensor for all the maxNumPeds in the frame
        params:
        grid_frame_data : A tensor of shape MNP x MNP x (GS**2)
        output_states : A list of tensors each of shape 1 x RNN_size of length MNP
        '''
        # Create a zero tensor of shape MNP x (GS**2) x RNN_size
        social_tensor = tf.zeros([self.args.maxNumPeds, self.grid_size*self.grid_size, self.rnn_size], name="social_tensor")
        
        
            
    """
    functions
    """
    def get_coef(self, output):
        # eq 20 -> 22 of Graves (2013)

        z = output
        # Split the output into 5 parts corresponding to means, std devs and corr
        z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(axis=1, num_or_size_splits=5, value=z)

        # The output must be exponentiated for the std devs
        z_sx = tf.exp(z_sx)
        z_sy = tf.exp(z_sy)
        # Tanh applied to keep it in the range [-1, 1]
        z_corr = tf.tanh(z_corr)

        return [z_mux, z_muy, z_sx, z_sy, z_corr]
    
    def tf_2d_normal(self, x, y, mux, muy, sx, sy, rho):
        '''
        Function that implements the PDF of a 2D normal distribution
        params:
        x : input x points
        y : input y points
        mux : mean of the distribution in x
        muy : mean of the distribution in y
        sx : std dev of the distribution in x
        sy : std dev of the distribution in y
        rho : Correlation factor of the distribution
        '''
        # eq 3 in the paper
        # and eq 24 & 25 in Graves (2013)
        # Calculate (x - mux) and (y-muy)
        normx = tf.subtract(x, mux)
        normy = tf.subtract(y, muy)
        sxsy = tf.multiply(sx, sy)
        z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) - 2*tf.div(tf.multiply(rho, tf.multiply(normx, normy)), sxsy)
        negRho = 1 - tf.square(rho)
        result = tf.exp(tf.div(-z, 2*negRho))
        denom = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negRho))
        result = tf.div(result, denom)
        self.result = result
        return result
    
    def get_lossfunc(self, z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
        '''
        Function to calculate given a 2D distribution over x and y, and target data
        of observed x and y points
        params:
        z_mux : mean of the distribution in x
        z_muy : mean of the distribution in y
        z_sx : std dev of the distribution in x
        z_sy : std dev of the distribution in y
        z_rho : Correlation factor of the distribution
        x_data : target x points
        y_data : target y points
        '''
        step = tf.constant(1e-3, dtype=tf.float32, shape=(1, 1))
        # Calculate the PDF of the data w.r.t to the distribution
        result0_1 = self.tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
        result0_2 = self.tf_2d_normal(tf.add(x_data, step), y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
        result0_3 = self.tf_2d_normal(x_data, tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)
        result0_4 = self.tf_2d_normal(tf.add(x_data, step), tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)
        result0 = tf.div(tf.add(tf.add(tf.add(result0_1, result0_2), result0_3), result0_4), tf.constant(4.0, dtype=tf.float32, shape=(1, 1)))
        #数值稳定性，加小系数求平均
        result0 = tf.multiply(tf.multiply(result0, step), step)
        # For numerical stability purposes
        epsilon = 1e-20
        result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability
        return tf.reduce_sum(result1)