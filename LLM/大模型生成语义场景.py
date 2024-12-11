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

# 初始化BERT的分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


client = OpenAI(
    base_url="https://api.gpts.vin/v1",
    api_key="sk-bCdM5TOGRboGu8ZK116e8170471848Bb947fA50bA583187e"
)

background = """
###指令### 您是一位自然语言处理的专家，请您帮助我们对一系列用户看过的电影属性进行总结,返回的结果为一段故事文本。总结出该用户喜欢的主要电影类型有哪几种，结合电影名称，为用户推理出一段背景文本，注意：请不要分段，字数限制在200字左右 
Q：['Scream (1996), 20-Dec-1996, Horror, Thriller',
 'Everyone Says I Love You (1996), 06-Dec-1996, Comedy, Musical, Romance',
 'Air Force One (1997), 01-Jan-1997, Action, Thriller',
 'Game, The (1997), 01-Jan-1997, Mystery, Thriller',
 'Conspiracy Theory (1997), 08-Aug-1997, Action, Mystery, Romance, Thriller',
 'Gattaca (1997), 01-Jan-1997, Drama, Sci - Fi, Thriller',
 'Evita (1996), 25-Dec-1996, Drama, Musical',
 'Seven Years in Tibet (1997), 01-Jan-1997, Drama, War',
 'I Know What You Did Last Summer (1997), 17-Oct-1997, Horror, Mystery, Thriller',
 "She's So Lovely (1997), 22-Aug-1997, Drama, Romance",
 'Contact (1997), 11-Jul-1997, Drama, Sci - Fi',
 'Cop Land (1997), 01-Jan-1997, Crime, Drama, Mystery',
 'Thousand Acres, A (1997), 01-Jan-1997, Drama',
 'Nixon (1995), 01-Jan-1995, Drama',
 'Sleepers (1996), 18-Oct-1996, Crime, Drama',
 'Mirror Has Two Faces, The (1996), 15-Nov-1996, Comedy, Romance',
 'Crucible, The (1996), 27-Nov-1996, Drama',
 'Toy Story (1995), 01-Jan-1995, Animation, Childrens, Comedy',
 'Zeus and Roxanne (1997), 10-Jan-1997, Childrens',
 'Eddie (1996), 31-May-1996, Comedy',
 'To Gillian on Her 37th Birthday (1996), 18-Oct-1996, Drama, Romance',
 'Independence Day (ID4) (1996), 03-Jul-1996, Action, Sci - Fi, War',
 'Ransom (1996), 08-Nov-1996, Drama, Thriller',
 'People vs. Larry Flynt, The (1996), 27-Dec-1996, Drama',
 'Dead Man Walking (1995), 01-Jan-1995, Drama',
 'Birdcage, The (1996), 08-Mar-1996, Comedy',
 'Sabrina (1995), 01-Jan-1995, Comedy, Romance',
 'Halloween: The Curse of Michael Myers (1995), 01-Jan-1995, Horror, Thriller',
 'Sense and Sensibility (1995), 01-Jan-1995, Drama, Romance',
 'Truth About Cats & Dogs, The (1996), 26-Apr-1996, Comedy, Romance',
 'Fargo (1996), 14-Feb-1997, Crime, Drama, Thriller',
 'Time to Kill, A (1996), 13-Jul-1996, Drama',
 "Mr. Holland's Opus (1995), 29-Jan-1996, Drama",
 'Phenomenon (1996), 29-Jun-1996, Drama, Romance',
 'That Thing You Do! (1996), 28-Sep-1996, Comedy',
 'Hunchback of Notre Dame, The (1996), 21-Jun-1996, Animation, Childrens, Musical',
 'Star Trek: First Contact (1996), 22-Nov-1996, Action, Adventure, Sci - Fi',
 'Mission: Impossible (1996), 22-May-1996, Action, Adventure, Mystery',
 'Primal Fear (1996), 30-Mar-1996, Drama, Thriller',
 'First Wives Club, The (1996), 14-Sep-1996, Comedy',
 'Jerry Maguire (1996), 13-Dec-1996, Drama, Romance',
 'Absolute Power (1997), 14-Feb-1997, Mystery, Thriller',
 'Twelve Monkeys (1995), 01-Jan-1995, Drama, Sci - Fi',
 '101 Dalmatians (1996), 27-Nov-1996, Childrens, Comedy',
 'Extreme Measures (1996), 27-Sep-1996, Drama, Thriller',
 "Preacher's Wife, The (1996), 13-Dec-1996, Drama",
 'White Squall (1996), 01-Jan-1996, Adventure, Drama',
 'One Fine Day (1996), 30-Nov-1996, Drama, Romance',
 'Craft, The (1996), 26-Apr-1996, Drama, Horror',
 'Emma (1996), 02-Aug-1996, Drama, Romance',
 'Nutty Professor, The (1996), 28-Jun-1996, Comedy, Fantasy, Romance, Sci - Fi',
 'Eraser (1996), 21-Jun-1996, Action, Thriller']
A：用户喜欢的电影类型主要包括喜剧、剧情、惊悚和科幻。从观看的电影列表来看，这位用户似乎对富有情感深度和复杂人物关系的电影情有独钟。他们欣赏从经典喜剧如《鸟笼》到深刻的剧情片如《死囚漫步》的各种类型。同时，对于探索人性和社会问题的影片如《一级恐惧》和《法网边缘》也表现出了浓厚的兴趣。此外，科幻电影如《接触》和《星际迷航：第一次接触》也受到了他们的青睐，显示出他们对于探索未知和未来世界的好奇心。
Q:['Shadow Conspiracy (1997), 31-Jan-1997, Thriller', 'Saint, The (1997), 14-Mar-1997, Action, Romance, Thriller',
 'G.I. Jane (1997), 01-Jan-1997, Action, Drama, War', 'Gattaca (1997), 01-Jan-1997, Drama, Sci - Fi, Thriller',
 'Fly Away Home (1996), 13-Sep-1996, Adventure, Childrens', 'Contact (1997), 11-Jul-1997, Drama, Sci - Fi',
 'U Turn (1997), 01-Jan-1997, Action, Crime, Mystery', 'Murder at 1600 (1997), 18-Apr-1997, Mystery, Thriller',
 'Liar Liar (1997), 21-Mar-1997, Comedy', "Devil's Own, The (1997), 26-Mar-1997, Action, Drama, Thriller, War",
 "McHale's Navy (1997), 18-Apr-1997, Comedy, War", 'Toy Story (1995), 01-Jan-1995, Animation, Childrens, Comedy',
 'Jerry Maguire (1996), 13-Dec-1996, Drama, Romance', 'Rock, The (1996), 07-Jun-1996, Action, Adventure, Thriller',
 'Heat (1995), 01-Jan-1995, Action, Crime, Thriller', 'Rumble in the Bronx (1995), 23-Feb-1996, Action, Adventure, Crime',
 'Long Kiss Goodnight, The (1996), 05-Oct-1996, Action, Thriller', 'Courage Under Fire (1996), 08-Mar-1996, Drama, War', "Jackie Chan's First Strike (1996), 10-Jan-1997, Action",
 'Star Trek: First Contact (1996), 22-Nov-1996, Action, Adventure, Sci - Fi', 'Twelve Monkeys (1995), 01-Jan-1995, Drama, Sci - Fi',
 'Spitfire Grill, The (1996), 06-Sep-1996, Drama', 'Fan, The (1996), 16-Aug-1996, Thriller',
 'Independence Day (ID4) (1996), 03-Jul-1996, Action, Sci - Fi, War', 'Othello (1995), 18-Dec-1995, Drama',
 'Swingers (1996), 18-Oct-1996, Comedy, Drama', 'Emma (1996), 02-Aug-1996, Drama, Romance',
 'Executive Decision (1996), 09-Mar-1996, Action, Thriller', 'People vs. Larry Flynt, The (1996), 27-Dec-1996, Drama',
 'First Wives Club, The (1996), 14-Sep-1996, Comedy',
 'Michael Collins (1996), 11-Oct-1996, Drama, War', 'My Fellow Americans (1996), 20-Dec-1996, Comedy',
 'Eddie (1996), 31-May-1996, Comedy', 'Ransom (1996), 08-Nov-1996, Drama, Thriller',
 'Multiplicity (1996), 12-Jul-1996, Comedy', 'Twister (1996), 10-May-1996, Action, Adventure, Thriller', 'Tin Cup (1996), 16-Aug-1996, Comedy, Romance',
 'Mars Attacks! (1996), 13-Dec-1996, Action, Comedy, Sci - Fi, War', 'Willy Wonka and the Chocolate Factory (1971), 01-Jan-1971, Adventure, Childrens, Comedy',
 'Broken Arrow (1996), 09-Feb-1996, Action, Thriller', 'Grumpier Old Men (1995), 01-Jan-1995, Comedy, Romance',
 'Sabrina (1995), 01-Jan-1995, Comedy, Romance', 'Phenomenon (1996), 29-Jun-1996, Drama, Romance',
 'Mission: Impossible (1996), 22-May-1996, Action, Adventure, Mystery', 'Truth About Cats & Dogs, The (1996), 26-Apr-1996, Comedy, Romance',
 'Sense and Sensibility (1995), 01-Jan-1995, Drama, Romance', 'Richard III (1995), 22-Jan-1996, Drama, War',
 'Up Close and Personal (1996), 01-Mar-1996, Drama, Romance', 'James and the Giant Peach (1996), 12-Apr-1996, Animation, Childrens, Musical',
 'Down Periscope (1996), 01-Mar-1996, Comedy', 'Frighteners, The (1996), 19-Jul-1996, Comedy, Horror',
 'Michael (1996), 25-Dec-1996, Comedy, Romance', 'Glimmer Man, The (1996), 04-Oct-1996, Action, Thriller',
 'Set It Off (1996), 25-Sep-1996, Action, Crime', 'Mirror Has Two Faces, The (1996), 15-Nov-1996, Comedy, Romance',
 'Phantom, The (1996), 07-Jun-1996, Adventure', 'Craft, The (1996), 26-Apr-1996, Drama, Horror', 'Nutty Professor, The (1996), 28-Jun-1996, Comedy, Fantasy, Romance, Sci - Fi',
 'Screamers (1995), 01-Jan-1995, Sci - Fi', 'Escape from L.A. (1996), 09-Aug-1996, Action, Adventure, Sci - Fi, Thriller',
 'Arrival, The (1996), 31-May-1996, Action, Sci - Fi, Thriller', 'Eraser (1996), 21-Jun-1996, Action, Thriller',
 'Absolute Power (1997), 14-Feb-1997, Mystery, Thriller', 'Dragonheart (1996), 31-May-1996, Action, Adventure, Fantasy',
 'High School High (1996), 25-Oct-1996, Comedy', 'Beverly Hills Ninja (1997), 17-Jan-1997, Action, Comedy', "Marvin's Room (1996), 18-Dec-1996, Drama",
 'Substitute, The (1996), 19-Apr-1996, Action', 'Space Jam (1996), 15-Nov-1996, Adventure, Animation, Childrens, Comedy, Fantasy',
 'Spy Hard (1996), 24-May-1996, Comedy', 'First Kid (1996), 30-Aug-1996, Childrens, Comedy', 'Bulletproof (1996), 06-Sep-1996, Action',
 'Sgt. Bilko (1996), 29-Mar-1996, Comedy']
 A:该用户喜欢的电影类型前4名为喜剧、剧情、惊悚和科幻。他们似乎偏爱那些能够提供情感共鸣和深刻思考的作品。从喜剧如《鸟笼》和《艾迪》到剧情片如《死囚漫步》和《尼克松》，用户对电影的选择显示了对复杂情感和人性探讨的偏好。同时，他们也喜欢充满悬疑和紧张气氛的惊悚片，例如《惊声尖叫》和《空军一号》。科幻电影如《千钧一发》和《接触》则可能满足了他们对未来和未知世界的好奇心。此外，用户对音乐剧和儿童电影也有所涉猎，表明他们对电影的欣赏是多元化的。
Q:['Swingers (1996), 18-Oct-1996, Comedy, Drama',
 'Jaws (1975), 01-Jan-1975, Action, Horror',
 'True Romance (1993), 01-Jan-1993, Action, Crime, Romance',
 'Lion King, The (1994), 01-Jan-1994, Animation, Childrens, Musical',
 'Clerks (1994), 01-Jan-1994, Comedy',
 'Aliens (1986), 01-Jan-1986, Action, Sci - Fi, Thriller, War',
 'Nightmare Before Christmas, The (1993), 01-Jan-1993, Childrens, Comedy, Musical',
 'Right Stuff, The (1983), 01-Jan-1983, Drama',
 "Bram Stoker's Dracula (1992), 01-Jan-1992, Horror, Romance",
 'Good, The Bad and The Ugly, The (1966), 01-Jan-1966, Action, Western',
 'When Harry Met Sally... (1989), 01-Jan-1989, Comedy, Romance',
 'Sting, The (1973), 01-Jan-1973, Comedy, Crime',
 'Maverick (1994), 01-Jan-1994, Action, Comedy, Western',
 'Three Colors: Red (1994), 01-Jan-1994, Drama',
 'Gone with the Wind (1939), 01-Jan-1939, Drama, Romance, War',
 'Terminator, The (1984), 01-Jan-1984, Action, Sci - Fi, Thriller',
 'Cape Fear (1991), 01-Jan-1991, Thriller',
 'Cinema Paradiso (1988), 01-Jan-1988, Comedy, Drama, Romance',
 'Platoon (1986), 01-Jan-1986, Drama, War',
 'Sling Blade (1996), 22-Nov-1996, Drama, Thriller',
 'Star Trek VI: The Undiscovered Country (1991), 01-Jan-1991, Action, Adventure, Sci - Fi',
 'Hot Shots! Part Deux (1993), 01-Jan-1993, Action, Comedy, War',
 'Ace Ventura: Pet Detective (1994), 01-Jan-1994, Comedy',
 'Maya Lin: A Strong Clear Vision (1994), 01-Jan-1994, Documentary',
 'Shining, The (1980), 01-Jan-1980, Horror',
 'Get Shorty (1995), 01-Jan-1995, Action, Comedy, Drama',
 'Field of Dreams (1989), 01-Jan-1989, Drama',
 'Abyss, The (1989), 01-Jan-1989, Action, Adventure, Sci - Fi, Thriller',
 'GoldenEye (1995), 01-Jan-1995, Action, Adventure, Thriller',
 'Akira (1988), 01-Jan-1988, Adventure, Animation, Sci - Fi, Thriller',
 'Firm, The (1993), 01-Jan-1993, Drama, Thriller',
 'Natural Born Killers (1994), 01-Jan-1994, Action, Thriller',
 'Mr. Smith Goes to Washington (1939), 01-Jan-1939, Drama',
 'Exotica (1994), 01-Jan-1994, Drama',
 'To Wong Foo, Thanks for Everything! Julie Newmar (1995), 01-Jan-1995, Comedy',
 'Fish Called Wanda, A (1988), 01-Jan-1988, Comedy',
 'Full Monty, The (1997), 01-Jan-1997, Comedy',
 'Haunted World of Edward D. Wood Jr., The (1995), 26-Apr-1996, Documentary',
 'Princess Bride, The (1987), 01-Jan-1987, Action, Adventure, Comedy, Romance',
 'M*A*S*H (1970), 01-Jan-1970, Comedy, War',
 'Star Trek III: The Search for Spock (1984), 01-Jan-1984, Action, Adventure, Sci - Fi',
 'Unforgiven (1992), 01-Jan-1992, Western',
 'Stargate (1994), 01-Jan-1994, Action, Adventure, Sci - Fi',
 'So I Married an Axe Murderer (1993), 01-Jan-1993, Comedy, Romance, Thriller',
 'Star Trek IV: The Voyage Home (1986), 01-Jan-1986, Action, Adventure, Sci - Fi',
 'On Golden Pond (1981), 01-Jan-1981, Drama',
 'Three Colors: White (1994), 01-Jan-1994, Drama',
 'Hunt for Red October, The (1990), 01-Jan-1990, Action, Thriller',
 'Priest (1994), 01-Jan-1994, Drama',
 'I.Q. (1994), 01-Jan-1994, Comedy, Romance',
 'Belle de jour (1967), 01-Jan-1967, Drama',
 'Sleeper (1973), 01-Jan-1973, Comedy, Sci - Fi',
 'Jurassic Park (1993), 01-Jan-1993, Action, Adventure, Sci - Fi',
 '20,000 Leagues Under the Sea (1954), 01-Jan-1954, Adventure, Childrens, Fantasy, Sci - Fi',
 'Mask, The (1994), 01-Jan-1994, Comedy, Crime, Fantasy',
 'Desperado (1995), 01-Jan-1995, Action, Romance, Thriller',
 "Weekend at Bernie's (1989), 01-Jan-1989, Comedy",
 'Nikita (La Femme Nikita) (1990), 01-Jan-1990, Thriller',
 'Horseman on the Roof, The (Hussard sur le toit, Le) (1995), 19-Apr-1996, Drama',
 'Sleepless in Seattle (1993), 01-Jan-1993, Comedy, Romance',
 'Sneakers (1992), 01-Jan-1992, Crime, Drama, Sci - Fi',
 'Disclosure (1994), 01-Jan-1994, Drama, Thriller',
 'Wizard of Oz, The (1939), 01-Jan-1939, Adventure, Childrens, Drama, Musical',
 'Indiana Jones and the Last Crusade (1989), 01-Jan-1989, Action, Adventure',
 'Patton (1970), 01-Jan-1970, Drama, War',
 'Cold Comfort Farm (1995), 23-Apr-1996, Comedy',
 'Phenomenon (1996), 29-Jun-1996, Drama, Romance',
 'Young Frankenstein (1974), 01-Jan-1974, Comedy, Horror',
 'Four Rooms (1995), 01-Jan-1995, Thriller',
 'Usual Suspects, The (1995), 14-Aug-1995, Crime, Thriller',
 'Quiz Show (1994), 01-Jan-1994, Drama',
 'Evil Dead II (1987), 01-Jan-1987, Action, Adventure, Comedy, Horror',
 'While You Were Sleeping (1995), 01-Jan-1995, Comedy, Romance',
 'Net, The (1995), 01-Jan-1995, Sci - Fi, Thriller',
 'Last of the Mohicans, The (1992), 01-Jan-1992, Action, Romance, War',
 "Carlito's Way (1993), 01-Jan-1993, Crime, Drama",
 'Die Hard 2 (1990), 01-Jan-1990, Action, Thriller',
 'Young Guns (1988), 01-Jan-1988, Action, Comedy, Western',
 'Love Bug, The (1969), 01-Jan-1969, Childrens, Comedy',
 'Brother Minister: The Assassination of Malcolm X (1994), 01-Jan-1994, Documentary',
 'Legends of the Fall (1994), 01-Jan-1994, Drama, Romance, War, Western',
 'Outbreak (1995), 01-Jan-1995, Action, Drama, Thriller',
 'Dolores Claiborne (1994), 01-Jan-1994, Drama, Thriller',
 'Searching for Bobby Fischer (1993), 01-Jan-1993, Drama',
 'Fargo (1996), 14-Feb-1997, Crime, Drama, Thriller',
 'Dead Man Walking (1995), 01-Jan-1995, Drama',
 'Star Trek: The Wrath of Khan (1982), 01-Jan-1982, Action, Adventure, Sci - Fi',
 "Monty Python's Life of Brian (1979), 01-Jan-1979, Comedy",
 'Remains of the Day, The (1993), 01-Jan-1993, Drama',
 'Wrong Trousers, The (1993), 01-Jan-1993, Animation, Comedy',
 'French Twist (Gazon maudit) (1995), 01-Jan-1995, Comedy, Romance',
 '12 Angry Men (1957), 01-Jan-1957, Drama',
 'Star Trek: First Contact (1996), 22-Nov-1996, Action, Adventure, Sci - Fi',
 'Contact (1997), 11-Jul-1997, Drama, Sci - Fi']
A:该用户喜欢的电影类型的前4名为喜剧、剧情、动作、和科幻。从《大白鲨》和《异形》这样的惊悚动作片，到《狮子王》和《圣诞夜惊魂》这样的动画音乐剧，再到《星球大战》和《星际迷航》系列的科幻冒险。这位用户似乎对电影史上的里程碑作品和不同流派的混合体都有着浓厚的兴趣，他们的观影清单显示了对电影多样性的欣赏，无论是战争片如《乱世佳人》和《现代启示录》，还是黑色喜剧如《疯狂原始人》和《公主新娘》，或是探讨社会问题的纪录片如《玛雅·林：清晰的强烈视觉》。这种广泛的电影品味表明用户对电影艺术的热爱和对不同故事叙述方式的探索。
"""


background_mv ="""
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


def loadusers():
    file_path = 'u.user'
    table = ['UserID', 'Age', 'Gender', 'Occupation']
    # UserID::Gender::Age::Occupation::Zip-code

    data = np.loadtxt(file_path, dtype=str)
    dat_set = []
    for dat in data:
        u_dat = dat.split("|")[:-1]
        dat_set.append(u_dat)
    # print(dat_set)

    data_set = np.array(dat_set)
    dat_line1 = data_set[:, 0]  # user
    dat_line2 = np.where(data_set[:, 2] == 'F', 0, 1)
    dat_line3 = []

    llm_user = []

    for age in data_set[:, 1]:
        if int(age) > 0 and int(age) < 10:
            dat_line3.append('0')
        elif int(age) >= 10 and int(age) < 20:
            dat_line3.append('1')
        elif int(age) >= 20 and int(age) < 30:
            dat_line3.append('2')
        elif int(age) >= 30 and int(age) < 40:
            dat_line3.append('3')
        elif int(age) >= 40 and int(age) < 50:
            dat_line3.append('4')
        elif int(age) >= 50 and int(age) < 60:
            dat_line3.append('5')
        elif int(age) >= 60:
            dat_line3.append('6')
    dat_line3 = np.array(dat_line3)
    dat_line4 = data_set[:, 3]
    # 编码
    dat_line4_map = {j: i for i, j in enumerate(set(dat_line4))}
    idx_dat_line4 = np.array(list(map(dat_line4_map.get, dat_line4)))

    data_all = np.concatenate([np.reshape(dat_line1, [-1, 1]), np.reshape(dat_line2, [-1, 1]),
                               np.reshape(dat_line3, [-1, 1]), np.reshape(idx_dat_line4, [-1, 1])], axis=1).astype(int)

    np.savetxt('kh/users_attributes.txt', data_all, fmt='%d', delimiter=',', encoding='utf-8')

    with open('kh/movies_llm.txt', 'w', encoding='utf-8') as file:
        print('save...')
        file.write(','.join(table) + '\n')
        for r in llm_user:
            file.write(','.join(r) + '\n')
        file.close()

    return data_set


def loadmovies():
    file_path = 'u_item.txt'
    table = ['MovieID', 'Title', 'Genres']

    mv_type = ['unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
               'Fantasy',
               'Film - Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci - Fi', 'Thriller', 'War', 'Western']
    mv_att = []
    mv_llm = []

    # 以二进制模式打开文件，读取一部分数据用于编码检测
    # with open(file_path, 'rb') as f:
    #     raw_data = f.read(32)  # 读取32字节足以进行编码检测
    #     print(raw_data)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            mv_att.append([])
            mv_llm.append([])
            row = line.replace("\n", "").split('|')
            mv_id = row[0]
            mv_att[-1].append(str(mv_id))
            mv_llm[-1].append(str(mv_id))
            mv_name = row[1]
            mv_llm[-1].append(str(mv_name))
            mv_date = row[2]
            mv_llm[-1].append(str(mv_date))
            ttype = row[-19:]
            for iddx, ty in enumerate(ttype):
                if ty == '1':
                    mv_att[-1].append(str(iddx))  # 电影 + 属性编号
                    mv_llm[-1].append(str(mv_type[iddx]))

    with open('kh/movies_attributes.txt', 'w', encoding='utf-8') as file:
        print('save...')
        # file.write(','.join(table) + '\n')
        for r in mv_att:
            file.write(','.join(r) + '\n')
        file.close()

    with open('kh/movies_llm.txt', 'w', encoding='utf-8') as file:
        print('save...')
        # file.write(','.join(table) + '\n')
        for r in mv_llm:
            file.write(','.join(r) + '\n')
        file.close()

    return mv_llm


def loadratings(user_att, movie_att):  # 按照时间划分
    file_path = 'u.data'
    dat_set = np.loadtxt(file_path, dtype=int, delimiter='\t')

    indices = np.argsort(dat_set[:, 3])
    sorted_dat_set = dat_set[indices]

    times = 10
    deate = int(len(sorted_dat_set) / times)

    timestamps = sorted_dat_set[:, 3]

    # 生成10个划分点
    num_partitions = 10
    partition_points = np.linspace(0, len(timestamps), num_partitions + 1).astype(int)

    # 初始化一个空列表来存储划分后的数组
    partitioned_data = []

    # 按照划分点将数组分割成10份
    for i in range(len(partition_points) - 1):
        start_index = partition_points[i]
        end_index = partition_points[i + 1]
        selectss = sorted_dat_set[start_index:end_index]
        partitioned_data.append(selectss[selectss[:, 2] >= 3])
    # 打印结果
    # for i, partition in enumerate(partitioned_data):
    #     print(f"Partition {i + 1}: {partition}")
    ### 用户的个性化偏好

    u_dict_list = []
    for i, partition in enumerate(partitioned_data):
        u_m_dict = {}
        for edge in partition:
            if edge[0] not in u_m_dict.keys():
                u_m_dict[edge[0]] = [", ".join(movie_att[edge[1] - 1][1:])]
            else:
                u_m_dict[edge[0]].append(", ".join(movie_att[edge[1] - 1][1:]))
        u_dict_list.append(u_m_dict)

    ### 电影的个性化偏好
    m_dict_list = []
    for i, partition in enumerate(partitioned_data):
        m_u_dict = {}
        for edge in partition:
            if edge[1] not in m_u_dict.keys():
                m_u_dict[edge[1]] = [", ".join(user_att[edge[0] - 1][1:])]
            else:
                m_u_dict[edge[1]].append(", ".join(user_att[edge[0] - 1][1:]))
        m_dict_list.append(m_u_dict)
    return u_dict_list, m_dict_list

def LLM_to_txt(num, savefilename, dataset_, background):

    for idxx, dataset in enumerate(dataset_):
        ##############################

        feature = []

        ##############################
        print(idxx, savefilename.format(idxx))
        with open(savefilename.format(idxx), 'w', encoding='utf-8') as savefile:
            # savefile = csv.writer(savefile)

            for i in tqdm(range(num)):
                if i in dataset.keys():
                    # for dat in dataset:
                    ############# gpt ############
                    try:
                        raw = getdata(str(dataset[i][:20]), background)
                        tttime.sleep(0.05)

                        encoded_inputs = tokenizer.encode_plus(raw, truncation_strategy='longest_first', max_length=512,
                                                               pad_to_max_length=True, return_tensors="pt")
                        with torch.no_grad():
                            try:
                                outputs = bert_model(**encoded_inputs)
                            except:
                                outputs = bert_model(**encoded_inputs)
                        last_hidden_states = np.reshape(outputs[1].numpy(), [-1, 768])#.astype(str)
                        feature.append(last_hidden_states)
                        # savefile.writerow(last_hidden_states)
                    except:
                        # savefile.writerow(np.reshape(np.zeros(768), [-1, 768])) #.astype(str)
                        feature.append(np.reshape(np.zeros(768), [-1, 768]))
                        print(str(dataset[i]))
                else:
                    # savefile.writerow(np.reshape(np.zeros(768), [-1, 768])) #.astype(str)
                    feature.append(np.reshape(np.zeros(768), [-1, 768]))
        feature_arr = np.reshape(np.array(feature), [-1, 768])
        np.savetxt(savefilename.format(idxx), feature_arr, fmt='%.3f' , delimiter=',', encoding='utf-8')
    # 用于LLM
    # np.savetxt('ml-1m/train.txt', data_train, fmt='%d', delimiter=',', encoding='utf-8')
    # np.savetxt('ml-1m/val.txt', data_val, fmt='%d', delimiter=',', encoding='utf-8')
    # np.savetxt('ml-1m/test.txt', data_test, fmt='%d', delimiter=',', encoding='utf-8')
    # np.savetxt('ml-1m/ml-1m.txt', dat_set, fmt='%d', delimiter=',', encoding='utf-8')


if __name__ == '__main__':
    user_att = loadusers()
    movie_att = loadmovies()
    u_dict_list, m_dict_list = loadratings(user_att, movie_att)
    LLM_to_txt(num=943, savefilename='kh/_u_embed_{}.txt', dataset_=u_dict_list, background=background)
    print('user finished')
    LLM_to_txt(num=1682, savefilename='kh/_m_embed_{}.txt', dataset_=m_dict_list, background=background_mv)
    print('movies finished')
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
