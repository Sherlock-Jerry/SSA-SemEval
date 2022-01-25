def cnt(file, max_len, sp):
    with open('Farm_laws_annotated/'+file,'r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.split(sp)
            nowds = len(words)
            if nowds > max_len:
                max_len = nowds
    return max_len

max_src_len = 0
max_src_len = cnt("dev_bert.sent",max_src_len," ")
max_src_len = cnt("train_bert.sent",max_src_len," ")
max_src_len = cnt("test_bert.sent",max_src_len," ")

max_trg_len = 0
max_trg_len = cnt("dev_bert.pointer",max_trg_len,"|")
max_trg_len = cnt("train_bert.pointer",max_trg_len,"|")
max_trg_len = cnt("test_bert.pointer",max_trg_len,"|")

print(max_src_len)
print(max_trg_len)

