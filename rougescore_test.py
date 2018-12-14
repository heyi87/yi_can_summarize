from rouge import Rouge
import argparse
import codecs
import logging


class CalculateRougeScore(object):

    def __init__(self,args):
        self.decoded_file = args.decode_file
        self.ref_file = args.ref_file
        self.amazon_file = args.amazon_file
        self.amazon_output_file = args.amazon_output_file

        logging.info("decoded file {} \n ref_file {} \n amazon_file {} \n amazon_output_file {}")

    def clean_data(self, line):
        line = line.split('\n')[0].replace('output=','')
        line = line.lower().replace('[^a-zA-Z ]', "").strip().replace('\s+','\s')
        return line

    def load_doc(self, filename):
        # open the file as read only
        data = []
        data_org = []
        with open(filename) as fp:
            for line in fp:
                data.append(self.clean_data(line))
                data_org.append(line)

        return data, data_org

    def load_doc_json(self, filename):
        # open the file as read only
        data = []
        summary_data=[]
        with open(filename) as fp:
            for line in fp:
                data.append(eval(line))
                summary_data.append(eval(line)['summary'].strip())
        return data, summary_data

    def create_text(amazon_data, decoded, ref_org):
        return u"review: {} \n \t DECODED: {} \n \t REF: {} \n\n\n ".format(amazon_data, decoded, ref_org)

    def find_the_review(self, summary_data, ref):
        try:
            index = [i for i, x in enumerate(summary_data) if x == ref][0]
            return index
        except Exception as e:
            return None

    def write_output(self, output):
        with codecs.open(self.output_dir_file, "w", "utf-8") as fp:
            for item in output:
                fp.write("%s\n" % item)

    def calculate_rouge_score_find_good_results(self,score=0.2):

        decoded, decoded_org = self.load_doc(self.decode_file)
        ref, ref_org = self.load_doc(self.ref_file)

        if self.amazon_file != False:
            amazon_data, summary_data = self.load_doc_json(self.amazon_file)

        output=[]
        if len(decoded) != len(ref):
            ref = ref[0:len(decoded)]

        assert len(decoded) == len(ref_org)

        all_scores ,number_of_items_to_consider= [], len(decoded)

        for i in xrange(len(decoded)):
            rouge = Rouge()
            scores = rouge.get_scores(decoded[i], ref[i])
            all_scores.append(scores)

            if self.amazon != False:

                if scores[0]['rouge-1']['f']>score:

                    index = self.find_the_review(summary_data, ref_org[i].replace('\n','').replace('output=','').strip()) #find the matching reviews
                    if index is not None:
                        text = self.create_text(amazon_data[index]['reviewText'], decoded[i], ref_org[i])
                        logging.info("{}".format(text))
                        output.append(text)
                    else:
                        logging.info('summary not found')

            items_left = number_of_items_to_consider-i
            logging.info("almost_finished: {}".format(items_left))

        self.write_output(self.output_dir_file, output)

        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--decode_file", default="")
    parser.add_argument("-r", "--ref_file")
    parser.add_argument("-am", "--amazon_file", default=False)
    parser.add_argument("-ao", "--amazon_output_file", default=False)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')

    args = parser.parse_args()

    CalculateRougeScore(args)

