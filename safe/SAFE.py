
import binascii
import capstone
import requests
import numpy as np
import json
import os
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity

class InstructionsConverter:

    def __init__(self, json_i2id):
        f = open(json_i2id, 'r')
        self.i2id = json.load(f)
        f.close()

    def convert_to_ids(self, instructions_list):
        ret_array = []
        # For each instruction we add +1 to its ID because the first
        # element of the embedding matrix is zero
        for x in instructions_list:
            if x in self.i2id:
                ret_array.append(self.i2id[x] + 1)
            elif 'X_' in x:
                # print(str(x) + " is not a known x86 instruction")
                ret_array.append(self.i2id['X_UNK'] + 1)
            elif 'A_' in x:
                # print(str(x) + " is not a known arm instruction")
                ret_array.append(self.i2id['A_UNK'] + 1)
            else:
                # print("There is a problem " + str(x) + " does not appear to be an asm or arm instruction")
                ret_array.append(self.i2id['X_UNK'] + 1)
        return ret_array

# ------------------------------------------------------------------------------


class FunctionNormalizer:

    def __init__(self, max_instruction):
        self.max_instructions = max_instruction

    def normalize(self, f):
        f = np.asarray(f[0:self.max_instructions], dtype=np.int32)
        length = f.shape[0]
        if f.shape[0] < self.max_instructions:
            f = np.pad(f, (0, self.max_instructions - f.shape[0]), mode='constant')
        return f, length

    def normalize_function_pairs(self, pairs):
        lengths = []
        new_pairs = []
        for x in pairs:
            f0, len0 = self.normalize(x[0])
            f1, len1 = self.normalize(x[1])
            lengths.append((len0, len1))
            new_pairs.append((f0, f1))
        return new_pairs, lengths

    def normalize_functions(self, functions):
        lengths = []
        new_functions = []
        for f in functions:
            f, length = self.normalize(f)
            lengths.append(length)
            new_functions.append(f.tolist())
        return new_functions, lengths


# ------------------------------------------------------------------------------


class SAFE:

    def __init__(self, SERVING_URL):
        self.SERVING_URL = "http://35.233.53.43:8500/v1/models/safe:predict"

    def filter_memory_references(self, i, symbols, API):
        inst = "" + i.mnemonic
        for op in i.operands:
            if (op.type == 1):
                inst = inst + " " + i.reg_name(op.reg)
            elif (op.type == 2):
                imm = int(op.imm)
                symbol = 'liavetevistiliavetevistisullerivedelfiume...INANIINANI'
                if str(imm) in symbols:
                    symbol = str(symbols[str(imm)])
                if inst == 'call' and symbol in API:
                    inst = inst + " " + symbol
                elif (-int(5000) <= imm <= int(5000)):
                    inst = inst + " " + str(hex(op.imm))
                else:
                    inst = inst + " " + str('HIMM')
            elif (op.type == 3):
                mem = op.mem
                if (mem.base == 0):
                    r = "[" + "MEM" + "]"
                else:
                    r = '[' + str(i.reg_name(mem.base)) + "*" + str(mem.scale) + "+" + str(mem.disp) + ']'
                inst = inst + " " + r
            if (len(i.operands) > 1):
                inst = inst + ","
        if "," in inst:
            inst = inst[:-1]
        inst = inst.replace(" ", "_")
        return str(inst)

    def filter_asm_and_return_instruction_list(self, address, asm, symbols, info, API):

        if info.is_64bit():
            mode = capstone.CS_MODE_64
        elif info.is_32bit():
            mode = capstone.CS_MODE_32

        if info.procName == "arm":
            md = capstone.Cs(capstone.CS_ARCH_ARM, capstone.CS_MODE_ARM)
        elif info.procName == "metapc":
            md = capstone.Cs(capstone.CS_ARCH_X86, mode)


        print 'Processor: {}, {} bit'.format(info.procName, str(mode*8))
        # Result: Processor: mipsr, 32bit, big endian

        binary = binascii.unhexlify(asm)

        md.detail = True
        insns = []
        cap_insns = []
        for i in md.disasm(binary, address):
            insns.append(self.filter_memory_references(i, symbols, API))
            cap_insns.append(i)
        return insns

    def get_safe_embedding(self, fcn_asm, ida_info):
        insns = self.filter_asm_and_return_instruction_list(0, fcn_asm, [], ida_info, [])
        prepappend = 'X_'
        instructions = [prepappend + x for x in insns]
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        converter = InstructionsConverter(os.path.join(__location__, "word2id.json"))
        normalizer = FunctionNormalizer(150)
        converted = converter.convert_to_ids(instructions)
        instructions, lenghts = normalizer.normalize_functions([converted])
        payload = {"signature_name": "safe", "inputs": {"instruction": instructions, "lenghts": lenghts}}
        r = requests.post(self.SERVING_URL, data=json.dumps(payload))
        embeddings = json.loads(r.text)
        if "outputs" in embeddings:
            return json.dumps(embeddings["outputs"][0])
        else:
            raise ValueError("Something bad happened when computing SAFE embeddings")

    def get_safe_embeddings(self, fcns_asm, ida_info):
        fncs = []
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        converter = InstructionsConverter(os.path.join(__location__, "word2id.json"))
        for fcn_asm in fcns_asm:
            insns = self.filter_asm_and_return_instruction_list(0, fcn_asm, [], ida_info, [])
            prepappend = 'X_'
            instructions = [prepappend + x for x in insns]
            fncs.append(converter.convert_to_ids(instructions))
        normalizer = FunctionNormalizer(150)
        instructions, lenghts = normalizer.normalize_functions(fncs)
        payload = {"signature_name": "safe", "inputs": {"instruction": instructions, "lenghts": lenghts}}
        r = requests.post(self.SERVING_URL, data=json.dumps(payload))
        embeddings = json.loads(r.text)
        ret = []
        if "outputs" in embeddings:
            for emb in embeddings["outputs"]:
                ret.append(json.dumps(emb))
        else:
            raise ValueError("Something bad happened when computing SAFE embeddings")
        return ret


class ProgramEmbeddingMatrix:

    def __init__(self):
        self.embedding_matrix = np.zeros([0, 100])

    def add_embedding(self, embedding):
        try:
            embedding = json.loads(embedding)
            self.embedding_matrix = np.vstack((self.embedding_matrix, np.asmatrix(embedding)))
        except:
            pass

class SimilarityFinder:

    def __init__(self, program_1, program_2):
        self.program_1 = program_1
        self.program_2 = program_2

    def find_similar(self, threshold):
        #dot = np.tensordot(self.program_1.embedding_matrix, self.program_2.embedding_matrix.T, axes=1)
        dot = cosine_similarity(self.program_1.embedding_matrix, self.program_2.embedding_matrix)
        max = np.amax(dot, axis=1)
        argmax = np.argmax(dot, axis=1)
        similar = []
        score = []
        for i in range(0, dot.shape[0]):
            if max[i] > threshold:
                similar.append((i, argmax[i]))
                score.append(max[i])
                break
        return similar, score


if __name__ == '__main__':

    conn = sqlite3.connect("../db1.sqlite")
    conn.text_factory = str
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute('attach "../db2.sqlite" as diff')

    p1 = ProgramEmbeddingMatrix()
    p2 = ProgramEmbeddingMatrix()

    sql1 = "select f.safe_embedding safe_embedding from functions f"
    q = cur.execute(sql1)
    res1 = q.fetchall()
    for f in res1:
        p1.add_embedding(f["safe_embedding"])

    sql1 = "select safe_embedding safe_embedding from diff.functions"
    q = cur.execute(sql1)
    res2 = q.fetchall()
    for f in res2:
        p2.add_embedding(f["safe_embedding"])

    s = SimilarityFinder(p1, p2)
    sim = s.find_similar(0.5)
    print(sim)



