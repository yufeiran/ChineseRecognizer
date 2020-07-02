#图像识别的主程序
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import shutil
import sys
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def CNN_CHINSES(filepath):
    
    trainDataPath="train"
    testDataPath="test"
    predictPath="predict"
    predictDataNewPath="predict1"
    learning_rate = 0.0001
    imageSize = 80
    batch_size = 100
    filename=os.path.basename(filepath)
    shutil.copyfile(filepath,os.path.join(predictPath,"000000",filename))
    sess = tf.compat.v1.Session()

    #首先我们从图片里以字符串的形式来读出图片内容：
    with sess.as_default():
        import pickle
        f = open('char_dict','br')
        dict = pickle.load(f)
        def readAdata(preditc_path):
            image_files=[]
            label_arr=[]
            for image in os.listdir(preditc_path):
                    image_files.append(os.path.join(preditc_path,image))
                    label_arr.append(int('00000'))
            images = tf.convert_to_tensor(preditc_path)
            labels = tf.convert_to_tensor(label_arr)
            return images,labels


        def read_data(dataPath):    
            image_files=[]
            label_arr=[]
            for char in os.listdir(dataPath):
                for image in os.listdir(os.path.join(dataPath,char)):
                    image_files.append(os.path.join(dataPath,char,image))
                    label_arr.append(int(char))
            images = tf.convert_to_tensor(image_files)
            labels = tf.convert_to_tensor(label_arr)
            return images,labels
        #images,labels = read_data(trainDataPath)
        #images_eval,labels_eval = read_data(testDataPath)

        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_png(image_string,channels=1)
            image_resized = tf.image.resize_images(image_decoded, [imageSize, imageSize])
            return image_resized, label

        def train_input_fn(trainDataPath):
            images,labels = read_data(trainDataPath)
            dataset = tf.data.Dataset.from_tensor_slices((images,labels))
            dataset = dataset.shuffle(buffer_size=895035).repeat()
            dataset = dataset.map(_parse_function).batch(batch_size)
            return dataset.make_one_shot_iterator().get_next()

        def eval_input_fn(testDataPath):
            images_eval,labels_eval = read_data(testDataPath)
            dataset = tf.data.Dataset.from_tensor_slices((images_eval,labels_eval))
            dataset = dataset.shuffle(buffer_size=895035).repeat()
            dataset = dataset.map(_parse_function).batch(batch_size)
            return dataset.make_one_shot_iterator().get_next()

        def pred_input_fn(predDataPath):
            images_pred,labels_pred = readAdata(predDataPath)
            dataset = tf.data.Dataset.from_tensor_slices((images_pred,labels_pred))
            dataset = dataset.shuffle(buffer_size=895035).repeat()
            dataset = dataset.map(_parse_function).batch(batch_size)
            return dataset.make_one_shot_iterator().get_next()

        def cnn_model_fn(features, labels, mode):
            input_layer =tf.reshape(features,[-1,80,80,1])
            conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            name="conv1")
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2,name="pool1")
            conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            name="conv2")
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],strides=2,name="pool2")
            conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=256,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
            pool3_flat = tf.reshape(pool3, [-1, 10 * 10 * 256])
            dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

            dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

            logits = tf.layers.dense(inputs=dropout, units=3755,name="logits")
            predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
            eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
            return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        chart_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir="logs")
        #训练
        #chart_classifier.train(input_fn=lambda:train_input_fn(trainDataPath),steps=200000)
        #eval_results=chart_classifier.evaluate(input_fn=lambda:eval_input_fn(testDataPath))
        #print(eval_results)

        #识别
        index_to_char = {value:key for key,value in dict.items()}
        #for pred in chart_classifier.predict(input_fn=lambda:pred_input_fn(predictDataNewPath)):
            #print(index_to_char[pred["classes"]])
        for pred in chart_classifier.predict(input_fn=lambda:eval_input_fn(predictPath)):
            print(index_to_char[pred["classes"]])
            os.remove(os.path.join(predictPath,"000000",filename))
            return

if __name__=='__main__':
    CNN_CHINSES(sys.argv[1])