 # -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
# import chardet
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import json
import time as tttime
from datetime import time
import numpy as np
import torch
import csv

from transformers import BertTokenizer, BertModel



client = OpenAI(
    base_url="https://api.gpts.vin/v1",
    api_key="sk-QhFyFRgVbLd6LtD6PFuFzwhDfeCPcJj7hilKO26iL94srFbW"
)



background_mv = """
###指令### 您是一位自然语言处理的专家，请您帮助我们对一系列用户看过的电影属性进行总结,返回的结果为一段故事文本。总结出该用户喜欢的主要电影类型有哪几种，结合电影名称，为用户推理出一段背景文本，注意：请不要分段，字数限制在200字左右 
Q: ['30, F, artist', '21, M, student', '37, M, engineer', '57, M, engineer', '19, M, student', '30, F, none', '25, M, student', '19, M, student', '28, M, student', '26, M, writer', '20, M, none', '29, F, educator', '24, M, technician', '20, M, student', '22, M, student', '26, M, entertainment', '33, M, programmer', '38, M, engineer', '20, F, student', '26, M, educator', '28, M, engineer', '25, M, other', '40, F, writer', '33, M, programmer', '27, M, programmer', '26, M, executive', '36, M, educator', '21, M, salesman', '23, M, other', '17, M, student', '29, F, artist', '31, M, student', '34, M, educator', '33, F, other', '40, M, other', '31, F, educator', '39, M, engineer', '24, M, engineer', '31, M, marketing', '38, F, entertainment', '32, M, entertainment', '28, M, student', '21, M, technician', '28, M, student', '29, F, librarian', '63, M, programmer', '28, M, programmer', '20, M, student', '40, M, programmer', '32, M, scientist', '27, F, educator', '30, M, engineer', '50, M, programmer', '29, M, engineer']
A: 最喜好该电影的职业是学生和工程师, 该电影似乎更受男性观众的欢迎, 主要受众年龄范围集中在19至38岁之间，电影市场整体呈现年轻化的趋势。

Q: ['20, M, student', '21, M, student', '46, M, programmer', '7, M, student', '34, M, other', '33, M, programmer', '35, M, technician', '29, M, engineer', '33, F, other', '32, M, entertainment', '40, M, other', '40, F, writer', '21, M, technician', '20, M, student', '27, F, educator', '20, M, none']
A: 在这些电影观众中，学生和程序员是最喜好该电影的职业，这表明电影可能具有较高的教育价值和对技术的吸引力。该电影似乎更受男性观众的欢迎，主要受众年龄范围集中在20至40岁之间，电影市场整体呈现年轻化和多样化的趋势。
"""


def getdata(text, background):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": background},
            {"role": "user", "content": 'Q:' + text}
        ])
    return completion.choices[0].message.content


def LLM_to_txt(dataset_, background):
    for idxx, dataset in enumerate(dataset_):
        ##############################
        raw = getdata(str(dataset)[:50], background) # 如果过长的话 str(dataset)[:50]
        time.sleep(0.05)
        ##############################
        print(raw)


if __name__ == '__main__':
    ## 调试如下程序即可
    u_jy_demo = []
    u_jy_demo.append("['30, F, artist', '21, M, student', '37, M, engineer', '57, M, engineer', '19, M, student', '30, F, none', '25, M, student', '19, M, student', '28, M, student', '26, M, writer', '20, M, none', '29, F, educator', '24, M, technician', '20, M, student', '22, M, student', '26, M, entertainment', '33, M, programmer', '38, M, engineer', '20, F, student', '26, M, educator', '28, M, engineer', '25, M, other', '40, F, writer', '33, M, programmer', '27, M, programmer', '26, M, executive', '36, M, educator', '21, M, salesman', '23, M, other', '17, M, student', '29, F, artist', '31, M, student', '34, M, educator', '33, F, other', '40, M, other', '31, F, educator', '39, M, engineer', '24, M, engineer', '31, M, marketing', '38, F, entertainment', '32, M, entertainment', '28, M, student', '21, M, technician', '28, M, student', '29, F, librarian', '63, M, programmer', '28, M, programmer', '20, M, student', '40, M, programmer', '32, M, scientist', '27, F, educator', '30, M, engineer', '50, M, programmer', '29, M, engineer']")
    u_jy_demo.append("['20, M, student', '21, M, student', '37, M, engineer', '27, M, student', '19, M, student', '30, F, none', '25, M, student', '19, M, student', '28, M, student', '26, M, writer', '20, M, none', '29, F, educator', '24, M, technician', '20, M, student', '22, M, student', '26, M, entertainment', '33, M, programmer', '38, M, engineer', '20, F, student', '26, M, educator', '28, M, engineer', '25, M, other', '40, F, writer', '33, M, programmer', '27, M, programmer', '26, M, executive', '36, M, educator', '21, M, salesman', '23, M, other', '17, M, student', '29, F, artist', '31, M, student', '34, M, educator', '33, F, other', '40, M, other', '31, F, educator', '39, M, engineer', '24, M, engineer', '31, M, marketing', '38, F, entertainment', '32, M, entertainment', '28, M, student', '21, M, technician', '28, M, student', '29, F, librarian', '63, M, programmer', '28, M, programmer', '20, M, student', '40, M, programmer', '32, M, scientist', '27, F, educator', '30, M, engineer', '50, M, programmer', '29, M, engineer']")

    num_u = 1
    LLM_to_txt(dataset_=u_jy_demo, background=background_mv)
    print('u finished')

"""
可以将提示词设计分为三个部分，分别对应任务的不同方面。这样可以更清晰地组织信息，帮助用户更好地理解和执行任务。以下是三部分的详细划分：

### 第一部分：角色和背景信息

**角色设定**：
- 自然语言处理专家

**背景信息**：
- 需要对关系文本进行简要扩建，以生成关系的描述和可能涉及的实体类型。

**个人简介**：
- 你是一位在自然语言处理领域有着深厚经验的专家，擅长分析和描述实体之间的关系描述。

**技能**：
- 关系分析、文本编写、客观描述、字符数控制。

### 第二部分：任务目标和限制

**目标**：
- 为给定的关系文本生成一个简洁、客观的描述，并简单描述可能涉及的实体类型。

**限制条件**：
- 文本扩建必须客观，不包含主观意向，且长度不超过50个字符。

**输出格式**：
- 一段简洁的英文文本描述。

### 第三部分：工作流程和示例

**工作流程**：
1. **分析关系文本**：
   - 提取关键信息，理解关系的本质。
2. **生成描述**：
   - 编写客观的关系描述。
3. **确定实体类型**：
   - 描述可能涉及的实体类型。
4. **确保简洁**：
   - 确保描述简洁，符合字符限制。

**示例**：
Q：business/company/founders
A：The term "business/company/founders" describes the relationship between a company and its founders. A company is a business organization, and founders are the individuals or teams who create and lead the company, and whose vision and decisions are critical to the company's success.

Q：/people/person/place_lived
A：The term "/people/person/place_lived" describes a place where a person has lived. This relationship links the person to their place of residence, which may include a city, country, or specific address.

Q：editor
A：The term "editor" describes the role of the editor within an organization. The relationship involves a link between the organization (e.g., publishing house, news organization) and the editor who is responsible for editing and managing the content to ensure its quality and consistency.

Q：director
A：The term "director" typically denotes a position within an organization or entity. This role involves overseeing and guiding the activities and decisions of the organization towards its goals. Possible entity types involved could include corporations, nonprofit organizations, or educational institutions.

通过这三部分的划分，可以更清晰地呈现提示词的各个方面，帮助用户系统地理解和执行任务。

"""
