from tempfile import NamedTemporaryFile
from pyspark.sql import SparkSession
import re
import gzip
from pyspark import SparkContext
from pyspark.sql.types import StringType,StructField,DoubleType,StructType
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import os
from functools import partial
from summarization.glove.crawl.local_warc import WARCFile
import nltk
import urllib.request

def create_spark_session(app_name="SparkApplication"):

    memory = '1g'
    pyspark_submit_args = ' --driver-memory ' + memory + ' pyspark-shell'
    os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args
    SparkContext.setSystemProperty('spark.executor.memory', '1g')
    # SparkContext.setSystemProperty('spark.driver.maxResultSize', '25g')

    spark_session = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate()

    spark_session.sparkContext.setLogLevel("WARN")
    return spark_session


def main():
    session = create_spark_session()
    sc = session.sparkContext
    base_url = "https://commoncrawl.s3.amazonaws.com/"
    file_with_urls_path = "input/wet.paths"
    schema = StructType([
        StructField("Target Word", StringType(), True),
        StructField("Context Word", StringType(), True),
        StructField("Coocurrence Prob", DoubleType(), True)
    ])
    urls = [x.strip('\n') for x in open(file_with_urls_path)]
    context_size = 10
    distributed_download_process(session, sc, urls, schema, base_url, context_size)
    #sequential_download_process(session, sc, urls, schema, base_url, context_size)
    #local_process(session, sc, "input", context_size)


def distributed_download_process(session, sc, urls, schema, base_url, context_size):

    urlsRDD = sc.parallelize(urls)
    out_dir_name = ''
    for url in urls:
        out_dir_name += url.split("/")[-1]
    corpuses = urlsRDD.map(lambda url: download_data(base_url + url, out_dir_name))
    list_of_records = corpuses.flatMap(lambda corpus: [re.sub(r'[^\x00-\x7F]+', ' ', record.payload.read().decode("UTF-8")) for i, record in enumerate(corpus)], 10)
    coocurrence = list_of_records.flatMap(lambda record: map_coocurence(context_size, record))
    coor_count = coocurrence.reduceByKey(lambda x, y: x + y)
    result = coor_count.map(lambda x: [x[0][0], x[0][1], x[1]])
    DF = session.createDataFrame(result, schema)
    DF.write.csv(os.path.join('output', out_dir_name))
    sc.stop()


def sequential_download_process(session, sc, urls, schema, base_url, context_size):
    for url in urls:
        session = create_spark_session()
        sc = session.sparkContext
        filename = url.split("/")[-1]
        corpus = download_data(base_url + url, filename)
        listOfRecords = sc.parallelize([re.sub(r'[^\x00-\x7F]+', ' ', record.payload.read().decode("UTF-8")) for i, record in enumerate(corpus)], 10)
        coor = listOfRecords.flatMap(lambda record : map_coocurence(context_size, record))
        coor_count = coor.reduceByKey(lambda x, y: x + y)
        result = coor_count.map(lambda x: [x[0][0], x[0][1], x[1]])
        DF = session.createDataFrame(result, schema)
        DF.write.csv(os.path.join('output', filename))
        sc.stop()


def local_process(session, sc, input_directory, context_size):
    from os import listdir
    from os.path import isfile, join
    filenames = [filename for filename in listdir(input_directory) if isfile(join(input_directory, filename))]
    file_paths = ["input/CC-MAIN-20170423031158-00000-ip-10-145-167-34.ec2.internal.warc.wet.gz"]
    for file_path in file_paths:
        filename = file_path.split("/")[-1]
        corpus = WARCFile(fileobj=gzip.open(file_path))
        list_of_records = sc.parallelize(
            [re.sub(r'[^\x00-\x7F]+', ' ', record.payload.read().decode("UTF-8")) for i, record in enumerate(corpus)], 10)
        coor = list_of_records.flatMap(partial(map_coocurence, context_size))
        coor_count = coor.reduceByKey(lambda x, y: x + y)
        result = coor_count.map(lambda x: [x[0][0], x[0][1], x[1]])
        tempFile = NamedTemporaryFile(delete=True)
        tempFile.close()
        schema = StructType([
            StructField("Target Word", StringType(), True),
            StructField("Context Word", StringType(), True),
            StructField("Coocurrence Prob", DoubleType(), True)
        ])
        DF = session.createDataFrame(result, schema)

        #DF.show()
        DF.write.csv(os.path.join('output', filename))


def download_data(fileurl, filename):
    import tempfile, shutil
    f = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
    fname = f.name
    with urllib.request.urlopen(fileurl) as response, open(fname, 'wb') as out_file:
        print(fname)
        shutil.copyfileobj(response, out_file)
        return WARCFile(fileobj=gzip.open(fname))


def map_coocurence(context_size, data):
    coocurrence_list = []
    try:
        if detect(data) == 'en':
            region = nltk.word_tokenize(data)
            for l_context, word, r_context in _context_windows(region, context_size, context_size):
                if isWord(word):
                    for i, context_word in enumerate(l_context[::-1]):
                        if isWord(context_word):
                            coocurrence_list.append(((word, context_word), 1 / (i + 1)))
                    for i, context_word in enumerate(r_context):
                        if isWord(context_word):
                            coocurrence_list.append(((word, context_word), 1 / (i + 1)))
    except LangDetectException:
        return coocurrence_list
    return coocurrence_list


def isWord(target):
    return re.match("^[A-Za-z]*$", target)


def _context_windows(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = _window(region, start_index, i - 1)
        right_context = _window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def _window(region, start_index, end_index):
    """
    Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
    return selected_tokens


if __name__ == '__main__':
    main()