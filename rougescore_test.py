from rouge import Rouge
import argparse
import codecs
def clean_data(line):
    line = line.split('\n')[0].replace('output=','')
    line = line.lower().replace('[^a-zA-Z ]', "").strip().replace('\s+','\s')
    return line

def load_doc(filename):
    # open the file as read only
    data = []
    data_org = []
    with open(filename) as fp:
        for line in fp:
            data.append(clean_data(line))
            data_org.append(line)

    return data, data_org

def load_doc_json(filename):
    # open the file as read only
    data = []
    summary_data=[]
    with open(filename) as fp:
        for line in fp:
            data.append(eval(line))
            summary_data.append(eval(line)['summary'].strip())
    return data, summary_data

def calculate_rouge_score(decoded, ref):
    if len(decoded) != len(ref):
        ref = ref[0:len(decoded)]

    all_scores = []

    for i in xrange(len(decoded)):
        rouge = Rouge()
        scores = rouge.get_scores(decoded[i], ref[i])
        all_scores.append(scores)

def calculate_rouge_score_find_good_results(decoded, decoded_org, ref, ref_org):

    output=[]
    if len(decoded) != len(ref):
        ref = ref[0:len(decoded)]
    assert len(decoded) == len(ref_org)

    amazon_data,summary_data = load_doc_json('/Users/yihe/Downloads/amazon_books.json')

    all_scores = []
    number_of_items_to_consider=len(decoded)
    for i in xrange(len(decoded)):
        rouge = Rouge()
        scores = rouge.get_scores(decoded[i], ref[i])
        all_scores.append(scores)

        if scores[0]['rouge-1']['r']>0.2:

            index = find_the_review(summary_data, ref_org[i].replace('\n','').replace('output=','').strip())
            if index is not None:
                text = "review: {} \n \t DECODED: {} \n \t REF: {} \n\n\n ".format(amazon_data[index]['reviewText'], decoded[i], ref_org[i])
                print "review: {} \n decoded: {} \n ref: {} \n ".format(amazon_data[index]['reviewText'], decoded[i], ref_org[i])
                output.append(text)
            else:
                print 'summary not found'

        items_left = number_of_items_to_consider-i
        print "almost_finished: {}".format(items_left)

    with codecs.open('good_amazon.txt', "w", "utf-8") as fp:
        for item in output:
            fp.write("%s\n" % item)

    return None

def find_the_review(summary_data, ref):
    try:
        index = [i for i, x in enumerate(summary_data) if x == ref][0]
        return index
    except Exception as e:
        return None


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument("-d", "--decode_file")
    # parser.add_argument("-r", "--ref_file")

    decode_file ='/Users/yihe/Downloads/decode1544661843'
    ref_file = '/Users/yihe/Downloads/ref1544661843'

    decoded, decoded_org=load_doc(decode_file)
    ref, ref_org = load_doc(ref_file)

    calculate_rouge_score_find_good_results(decoded, decoded_org, ref,ref_org)

