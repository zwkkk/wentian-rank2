python3 wrapper.py
cp -r model/bert_base_hard ../submit/bert_submit
mv ../submit/bert_submit/bert_base_hard ../submit/bert_submit/model 
cd ../submit/bert_submit
tar zcvf foo.tar.gz doc_embedding query_embedding model