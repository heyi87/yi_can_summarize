import tensorflow as tf
import codecs

data= []
try:
    i=0
    with codecs.open('tensor_data.csv', "w", "utf-8") as fp:
        for each_event in tf.train.summary_iterator("/Users/yihe/Downloads/events.out.tfevents.1544572115.ip-172-31-2-221"):
            i+=1
            print i
            try:
                if each_event.summary.value[0].tag == u'running_avg_loss':
                    fp.write("%s\n" % "{}, {}, {}".format(each_event.step, each_event.summary.value[0].tag, each_event.summary.value[0].simple_value))

            except Exception as e:
                print e
            #try:
                #if each_event.summary.value is not []:
                 #   event=[each_event.wall_time, each_event.step, each_event.summary.value.tag, each_event.summary.value.simple_value]
             #   fp.write("%s\n" % each_event.wall_time, each_event.step, each_event.summary.value.tag, each_event.summary.value.simple_value)
            #except Exception as e:
            #    print e
except Exception as e:
    print e