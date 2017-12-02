
# coding: utf-8

# <b> (Kaggle) MNIST - TensorFlow - Softmax Regression </b>

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np

# read training data from CSV file 
data = pd.read_csv('/Users/hardikthaker/Downloads/train.csv')

print('data({0[0]},{0[1]})'.format(data.shape))
print (data.head())


# In[2]:


images = data.iloc[:,1:].values
images = images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)

print('images({0[0]},{0[1]})'.format(images.shape))


# In[3]:


# read test data from CSV file 
test_images = pd.read_csv('/Users/hardikthaker/Downloads/test.csv').values
test_images = test_images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)

print('test_images({0[0]},{0[1]})'.format(test_images.shape))


# In[4]:


image_size = images.shape[1]
print ('image_size => {0}'.format(image_size))

# in this case all images are square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))


# In[21]:


labels_flat = data.iloc[:, 0].values.ravel()

print('labels_flat({0})'.format(len(labels_flat)))


# In[23]:


labels_count = np.unique(labels_flat).shape[0]

print('labels_count => {0}'.format(labels_count))


# In[24]:


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

print('labels({0[0]},{0[1]})'.format(labels.shape))


# <b>Let's create our placeholder. It is going to store our input 'x'.</b>
#     <p>Here, 'None' means that it can be of any dimension. That means any number of input images.
#     Dimension 784 is because each image is of size 28*28 which is being flattened. So, total 784 pixel values.

# In[5]:


x = tf.placeholder(tf.float32,[None,784])


# <b> Let's create our Weights and biases varibales. </b>
# <p> Note that now we are using tensorflow variables. On running our algorithm, we will be changing weights and biases. <b>Variable</b> is a modifiable tensor. So let's initialize <b>W</b> and <b>b</b>
#     <p> We are initializing W and b with all 0s. It doesn't matter usually what are the initial values.

# In[6]:


W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# <b> Let's implement our model. </b>
# <p> y = Softmax(Wx + b). Softmax regression is used to convert our output(evidences) into probabilities.

# In[7]:


y = tf.nn.softmax(tf.matmul(x, W) + b)


# <b> Now we define cross-entropy </b>
#     <p> Above define 'y' is our prediction. We cross check it with our actual ground truth labels 'y\_' to get the error.
#     <p> So, let's define 'y\_'

# In[8]:


y_ = tf.placeholder(tf.float32,[None,10])


# cross-entropy

# In[9]:


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# <b> Perform Optimization using Gradient Descent to reduce the error</b>
#     <p> 0.5 is the learning rate.

# In[10]:


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# <b> Launch the model in an Interactive Session and run initializer to initialize the variables we created. </b>

# In[11]:


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
file_writer = tf.summary.FileWriter('/Users/hardikthaker/output', sess.graph)


# In[25]:


train_images = images[2000:]
train_labels = labels[2000:]


# <b> Let's run training </b>
# <p> For 500 steps

# In[27]:


for _ in range(500):
  batch_xs, batch_ys = train_images, train_labels
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# <p> Compare our output 'y' with 'y\_'. argmax() function which gives you the index of the highest entry in a tensor along some axis.

# In[ ]:


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


# <p> We calculate accuracy from the correct_prediction.

# In[ ]:


#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# <p> Print the Accuracy.

# In[ ]:


#print(sess.run(accuracy, feed_dict={x: test_images, y_: mnist.test.labels}))


# <b> Export our Predictions </b>

# In[29]:


prediction = tf.argmax(y,1)
predicted_labels = prediction.eval(feed_dict={x: test_images})
print(prediction.eval(feed_dict={x: test_images}))


# <p><b> Export CSV file for Kaggle submission.</b>
#     

# In[31]:


import numpy as np
np.savetxt('submission.csv', 
           np.c_[range(1,len(test_images)+1),predicted_labels], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')


# In[32]:


get_ipython().system('ls')

