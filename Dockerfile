FROM python:3


WORKDIR /root/PerformaceCurveClustering

RUN git clone https://gitee.com/baiyanquan/KnowledgeReasoning-PerformaceCurveClustering.git /root/PerformaceCurveClustering \
    && pip install --upgrade pip \
    && pip install tslearn -i https://pypi.tuna.tsinghua.edu.cn/simple\
    && pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple\
    && pip install flask_cors -i https://pypi.tuna.tsinghua.edu.cn/simple\
        && pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple\
        && pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple\
        && pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

CMD ["python", "/root/PerformaceCurveClustering/app.py"]