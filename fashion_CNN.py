import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import import_data


train_image, train_label, test_image, test_label=import_data.read_data()

n_input=28
n_class=10
image_size=28
max_epochs=12000
Learning_rate=1e-3
batch_size=550
regularization=1e-2
drop_out=0.95
display_step=100

seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

weight={

   'wc1':tf.Variable(tf.random_normal([5,5,1,32]),name="wc1"),
   'wc2':tf.Variable(tf.random_normal([5,5,32,64]),name="wc2"),
   'wd1':tf.Variable(tf.random_normal([(image_size//4)*(image_size//4)*64,256]), name="wd1"),
   'w_out':tf.Variable(tf.random_normal([256, n_class]), name="w_out")
   
}

bias={

    'bc1':tf.Variable(tf.random_normal([32]),name="bc1"),
    'bc2':tf.Variable(tf.random_normal([64]),name="bc2"),
    'bd1':tf.Variable(tf.random_normal([256]),name="bd1"),
    'b_out':tf.Variable(tf.random_normal([n_class]),name="b_out")

}



def add_l2_loss(weight, bias):
    tf.add_to_collection("loss",tf.nn.l2_loss(weight))
    tf.add_to_collection("loss",tf.nn.l2_loss(bias))


def Loss(pred,label):
    corss_entropy_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=label))
    reg_loss=tf.add_n(tf.get_collection("loss"))
    loss=corss_entropy_loss+regularization*reg_loss
    tf.summary.scalar("Regularization",reg_loss)
    tf.summary.scalar("Cross_Entropy_Loss",loss)

    return loss


def CNN(X,weight,bias):

    Con_V1=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X,weight['wc1'],strides=[1,1,1,1],padding='SAME'),bias['bc1']))
    Con_V1=tf.nn.max_pool(Con_V1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    add_l2_loss(weight['wc1'],bias['bc1'])
    Con_V1=tf.nn.dropout(Con_V1,drop_out)
    tf.summary.histogram("wc1", weight['wc1'])
    tf.summary.histogram("bc1", bias['bc1'])

    
    Con_V2=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(Con_V1,weight['wc2'],strides=[1,1,1,1],padding='SAME'),bias['bc2']))
    Con_V2=tf.nn.max_pool(Con_V2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    add_l2_loss(weight['wc2'],bias['bc2'])
    Con_V2=tf.nn.dropout(Con_V2,drop_out)
    tf.summary.histogram("wc2", weight['wc2'])
    tf.summary.histogram("bc2", bias['bc2'])
    
    dense_layer=tf.reshape(Con_V2,[-1,weight['wd1'].get_shape().as_list()[0]])
    hidden_layer=tf.nn.relu(tf.add(tf.matmul(dense_layer,weight['wd1']),bias['bd1']))
    add_l2_loss(weight['wd1'],bias['bd1'])
    tf.summary.histogram("wd1", weight['wd1'])
    tf.summary.histogram("bd1", bias['bd1'])
    
    pred=tf.nn.softmax(tf.add(tf.matmul(hidden_layer,weight['w_out']),bias['b_out']),name="output")
    add_l2_loss(weight['w_out'],bias['b_out'])
    tf.summary.histogram("w_out", weight['w_out'])
    tf.summary.histogram("b_out", bias['b_out'])
    
    return pred




def main(argv=None):

    X=tf.placeholder(tf.float32, [None,28,28,1], name="input")
    Y=tf.placeholder(tf.float32, [None,n_class])
    
    pred=CNN(X,weight,bias)
    loss=Loss(pred,Y)
    train_op=tf.train.AdamOptimizer(learning_rate=Learning_rate).minimize(loss)
    correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar("accuracy",accuracy)
    

    init_op=tf.global_variables_initializer()
    summary_op=tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init_op)
        writer=tf.summary.FileWriter("output",sess.graph)


        for epochs in range(max_epochs):
            index=np.random.choice(train_image.shape[0],size=batch_size)
            batch_image=train_image[index]
            batch_label=train_label[index]
            sess.run([train_op],feed_dict={X:batch_image.reshape(-1,28,28,1),Y:batch_label})


            if epochs % display_step==0:
                train_loss,summary=sess.run([loss,summary_op],feed_dict={X:batch_image.reshape(-1,28,28,1),Y:batch_label})
                test_accuarcy,summary=sess.run([accuracy,summary_op],feed_dict={X:test_image.reshape(-1,28,28,1),Y:test_label})
                writer.add_summary(summary,global_step=epochs)
                print("Training Loss: {:.9f}  Test Accuarcy: {}".format(train_loss,test_accuarcy))
        



        print("Finish Training......")















if  __name__=="__main__":
    tf.app.run()