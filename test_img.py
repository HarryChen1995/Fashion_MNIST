import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import import_data



train_image, train_label, test_image, test_label=import_data.read_data()



def label(x):
    if np.argmax(x) == 0:
        return "T-shirt"
    if np.argmax(x) == 1:
        return "Trouser"
    if np.argmax(x) == 2:
        return "Pullover"
    if np.argmax(x) == 3:
        return "Dress"
    if np.argmax(x) == 4:
        return "Coat"
    if np.argmax(x) == 5:
        return "Sandal"
    if np.argmax(x) == 6:
        return "Shirt"
    if np.argmax(x) == 7:
        return "Sneaker"
    if np.argmax(x) == 8:
        return "Bag"
    if np.argmax(x) == 9:
        return "Ankle boot"


with tf.Session() as sess:
    saver=tf.train.import_meta_graph("output/model.ckpt-6000.meta")
    saver.restore(sess,"output/model.ckpt-6000")
    Graph=tf.get_default_graph()


    X = Graph.get_tensor_by_name("input:0")
    Y = Graph.get_tensor_by_name("output:0")
    j=1
    for i in range(10,19):
       
        plt.subplot(3,3,j)
        plt.imshow(np.resize(test_image[i],(28,28)), cmap="gray")
        Label=sess.run(Y, feed_dict={X:test_image[i].reshape(-1,28,28,1)})
        plt.title("CNN Prediction: "+label(Label)+" / True Class: "+label(test_label[i]), fontsize=11)
        j = j + 1
    plt.show()
