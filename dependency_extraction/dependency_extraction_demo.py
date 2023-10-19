import stanza
import time

start_time = time.time()
stanza.download('en')
end_time = time.time()
print("Download time: ", end_time - start_time)

nlp = stanza.Pipeline()

text = "Plaintiff United States Securities and Exchange Commission (the “Commission”), 100 F Street, N.E., Washington, DC 20549, alleges as follows against Defendant Yu-Cheng Lin, also known as Believe Lin, whose last known contact information is as follows: Residential Address, No. 53, XinYuan Road, QiPan Village, KuKung Township, YunLin County, Taiwan 646; Postal Address, 9F No. 32, Lane 22, GuangFu South Road, SongShan Dist., Taipei City 105, Taiwan."

start_time = time.time()
doc = nlp(text)
end_time = time.time()
print("NLP time: ", end_time - start_time)

doc.sentences[0].print_dependencies()
