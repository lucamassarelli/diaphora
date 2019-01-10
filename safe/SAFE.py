
import binascii
import capstone
import requests
import numpy as np
import json
import os

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
        f = np.asarray(f[0:self.max_instructions])
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

    def filter_asm_and_return_instruction_list(self, address, asm, symbols, arch, mode, API):
        binary = binascii.unhexlify(asm)
        if arch == capstone.CS_ARCH_ARM:
            md = capstone.Cs(capstone.CS_ARCH_ARM, capstone.CS_MODE_ARM)
        else:
            md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = True
        insns = []
        cap_insns = []
        for i in md.disasm(binary, address):
            insns.append(self.filter_memory_references(i, symbols, API))
            cap_insns.append(i)
        return insns

    def get_safe_embedding(self, fcn_asm):
        insns = self.filter_asm_and_return_instruction_list(0, fcn_asm, [], capstone.CS_ARCH_X86, capstone.CS_MODE_64, [])
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
