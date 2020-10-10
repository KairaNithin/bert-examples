
import tensorflow as tf
import tensorflow_hub as hub

data = {"data":
 [
     {"title": "Project Apollo",
      "paragraphs": [
          {"context":"The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which accomplished landing the first humans on the Moon from 1969 to 1972. First conceived during Dwight D. Eisenhower's administration as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space, Apollo was later dedicated to President John F. Kennedy's national goal of landing a man on the Moon and returning him safely to the Earth by the end of the 1960s, which he proposed in a May 25, 1961, address to Congress. Project Mercury was followed by the two-man Project Gemini. The first manned flight of Apollo was in 1968. Apollo ran from 1961 to 1972, and was supported by the two man Gemini program which ran concurrently with it from 1962 to 1966. Gemini missions developed some of the space travel techniques that were necessary for the success of the Apollo missions. Apollo used Saturn family rockets as launch vehicles. Apollo/Saturn vehicles were also used for an Apollo Applications Program, which consisted of Skylab, a space station that supported three manned missions in 1973-74, and the Apollo-Soyuz Test Project, a joint Earth orbit mission with the Soviet Union in 1975.",
           "qas": [
               { "question": "What project put the first Americans into space?",
                 "id": "Q1"
               },
               { "question": "What program was created to carry out these projects and missions?",
                 "id": "Q2"
               },
               { "question": "What year did the first manned Apollo flight occur?",
                 "id": "Q3"
               },
               { "question": "What President is credited with the original notion of putting Americans in space?",
                 "id": "Q4"
               },
               { "question": "Who did the U.S. collaborate with on an Earth orbit mission in 1975?",
                 "id": "Q5"
               },
               { "question": "How long did Project Apollo run?",
                 "id": "Q6"
               },
               { "question": "What program helped develop space travel techniques that Project Apollo used?",
                 "id": "Q7"
               },
               {"question": "What space station supported three manned missions in 1973-1974?",
                 "id": "Q8"
               }
]}]}]}
initializer = tf.keras.initializers.TruncatedNormal(
        stddev=0.02)
max_seq_length = 128
input_word_ids = tf.keras.layers.Input(
    shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
input_mask = tf.keras.layers.Input(
    shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
input_type_ids = tf.keras.layers.Input(
    shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
core_model = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)
pooled_output, sequence_output = core_model(
    [input_word_ids, input_mask, input_type_ids])


intermediate_logits = tf.keras.layers.Dense(2,kernel_initializer=initializer,name='predictions/transform/logits')(sequence_output)
start_logits, end_logits = (tf.keras.layers.Lambda(tf.unstack(tf.transpose(tf.identity, [2, 0, 1])))(intermediate_logits))
start_predictions = tf.keras.layers.Activation(tf.nn.log_softmax)(start_logits)
end_predictions = tf.keras.layers.Activation(tf.nn.log_softmax)(end_logits)
start_logits = tf.keras.layers.Lambda(tf.identity, name='start_positions')(start_logits)
end_logits = tf.keras.layers.Lambda(tf.identity, name='end_positions')(end_logits)
logits = [start_logits, end_logits]
bert_encoder = tf.keras.Model(
    inputs={
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids,
    },
    outputs=[logits],
    name='core_model')