import os
import codecs
import logging
import argparse

def load_doc(filename):
    # open the file as read only
    data = []
    with open(filename) as fp:
        for line in fp:
            data.append(eval(line))

    return data

def clean_data(line):
    return line.lower().replace('[^a-zA-Z ]', "").strip().replace('\s+','\s')

def get_data_from_json(line):
    return clean_data(line['summary']), clean_data(line['reviewText'])

def main(args):

    object_directory = args.object_directory
    output_dir = args.output_dir

    object = load_doc(object_directory)
    if os.path.isdir(output_dir) is False:
        os.mkdir(output_dir)

    number_of_files = len(object)
    number_of_files_completed = 0
    number_of_files_rejected = 0

    for each_review in object:
        summary, review = get_data_from_json(each_review)
        if len(review)>2 and len(summary)>2:

            file_output = u"{}\n@highlight\n{}".format(review, summary)

            output_file = os.path.join(output_dir, '{}.story'.format(each_review['asin']))

            with codecs.open(output_file, "w", "utf-8") as fp:
                fp.write(file_output)

            number_of_files_completed +=1

            logging.info("output_file completed \n Number of files left: {}".format(number_of_files-number_of_files_completed-number_of_files_rejected))
        else:
            number_of_files_rejected+=1
            logging.info("number of files rejected {}".format(number_of_files_rejected))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--object_directory")
    parser.add_argument("-o", "--output_dir")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    main(args)

