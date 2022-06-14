from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.tokenizers import Tokenizer
import jieba
import numpy as np
import onnxruntime 

jieba.initialize()

dict_path = './vocab/vocab.txt'

tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)

encoder_onnx = './onnx/encoder.onnx'
decoder_onnx = './onnx/decoder.onnx'

encoder_session = onnxruntime.InferenceSession(encoder_onnx, providers=['CPUExecutionProvider'])
decoder_session = onnxruntime.InferenceSession(decoder_onnx, providers=['CPUExecutionProvider'])

max_c_len = 1024
max_t_len = 30


class AutoTitleOnnx(AutoRegressiveDecoder):
    """
    seq2seq解码器
    """

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        c_encoded = inputs[0]
        return decoder_session.run("", {"Decoder-Input-Token": np.float32(output_ids), "Input-Context": np.float32(c_encoded)})[0]

    def generate(self, text, topk=1):
        c_token_ids, _ = tokenizer.encode(text, maxlen=max_c_len)
        c_encoded = encoder_session.run("", {"Encoder-Input-Token": [c_token_ids]})[0][0]
        output_ids = self.beam_search([c_encoded], topk=topk)  # 基于 beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitleOnnx(
    start_id=tokenizer._token_start_id,
    end_id=tokenizer._token_end_id,
    maxlen=max_t_len,
)

autotitle.generate("21日，习近平总书记在陕西省安康市平利县老县镇考察调研时，对镇中心小学的孩子们说：“现在孩子普遍眼镜化，这是我的隐忧。还有身体的健康程度，由于体育锻炼少，有所下降。文明其精神，野蛮其体魄，我说的‘野蛮其体魄’就是强身健体。”拥有强健体魄是青少年健康发展的基础。健康不是一切，但没有健康就没有一切。近年来各省区市每年的青少年体质监测显示，近视率和肥胖率居高不下，青少年体质状况不容乐观。加强体育锻炼，增强青少年体质，帮助学生在体育锻炼中享受乐趣、增强体质、健全人格、锤炼意志，既是百年大计，又是当务之急，必须引起全社会的高度重视。让体育陪伴青少年成长。少年强、青年强则中国强。中华民族的伟大复兴离不开一个肌体强健的民族。增强青少年体质、促进青少年健康成长，是关系国家和民族未来的大事。近年来，国家加大了学校体育的考核督查力度，多个省份出台新政，要求中小学校确保学生每日体育锻炼时间，这些都是利好消息。要形成社会共识，多管齐下增强青少年体质，把“体育课”补起来、让学校有因地制宜选择体育项目开展教学活动的空间，让学生有从事体育活动、激发运动潜能的时间。要把青少年身心健康摆在教育的首要位置，让青少年体魄强壮起来，让体育成为青少年成长的助推器。青少年身心健康、体魄强健，是国家综合实力的重要体现。让青少年“野蛮”一点、健康成长，全社会都要重视起来、行动起来，共同呵护一双双明亮眼睛，共同培育好民族的未来和希望。")
