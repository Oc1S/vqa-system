from os.path import join
from os import environ
import tensorflow as tf
import pickle
import numpy as np
import re
import apis.vqa.vis_lstm_model as vis_lstm_model
import apis.vqa.utils as utils


def pred(
    image_file,
    question="What color is the signal",
    model_path=join("models", "model.ckpt"),
    data_dir=join("apis", "vqa", "data"),
    num_lstm_layers=2,
    fc7_feature_length=4096,
    rnn_size=512,
    embedding_size=512,
    word_emb_dropout=0.5,
    image_dropout=0.5,
):
    environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    vocab_data = pickle.load(open(join(data_dir, "vocab_file2.pkl"), "rb"))
    fc7_features = utils.extract_fc7_features(
        image_file, join(data_dir, "vgg16.tfmodel")
    )

    model_options = {
        "num_lstm_layers": num_lstm_layers,
        "rnn_size": rnn_size,
        "embedding_size": embedding_size,
        "word_emb_dropout": word_emb_dropout,
        "image_dropout": image_dropout,
        "fc7_feature_length": fc7_feature_length,
        "lstm_steps": vocab_data["max_question_length"] + 1,
        "q_vocab_size": len(vocab_data["question_vocab"]),
        "ans_vocab_size": len(vocab_data["answer_vocab"]),
    }

    question_vocab = vocab_data["question_vocab"]
    word_regex = re.compile(r"\w+")
    question_ids = np.zeros((1, vocab_data["max_question_length"]), dtype="int32")
    question_words = re.findall(word_regex, question)
    base = vocab_data["max_question_length"] - len(question_words)
    for i in range(0, len(question_words)):
        if question_words[i] in question_vocab:
            question_ids[0][base + i] = question_vocab[question_words[i]]
        else:
            question_ids[0][base + i] = question_vocab["UNK"]
    ans_map = {
        vocab_data["answer_vocab"][ans]: ans for ans in vocab_data["answer_vocab"]
    }
    model = vis_lstm_model.Vis_lstm_model(model_options)
    input_tensors, t_prediction, t_ans_probab = model.build_generator()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    saver.restore(sess, join(data_dir, model_path))

    pred, answer_probab = sess.run(
        [t_prediction, t_ans_probab],
        feed_dict={
            input_tensors["fc7"]: fc7_features,
            input_tensors["sentence"]: question_ids,
        },
    )

    answer_probab_tuples = [
        (-answer_probab[0][idx], idx) for idx in range(len(answer_probab[0]))
    ]
    answer_probab_tuples.sort()
    # print("Top Answers")
    # for i in range(5):
    #     print(ans_map[answer_probab_tuples[i][1]])
    return [ans_map[tuple[1]] for tuple in answer_probab_tuples[:5]]
