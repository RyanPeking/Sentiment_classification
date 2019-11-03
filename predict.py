import pickle
from keras.preprocessing import sequence
from TextCNN.TextCNN import get_model
import jieba
import numpy as np
import os


maxlen = 260
tokener_filepath = os.path.join(os.path.abspath('./'), 'dataset', 'tokenizer.pickle')

# 位置
# 交通是否便利
location_traffic_convenience = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_location_traffic_convenience_03.hdf5')
# 距离商圈远近
location_distance_from_business_district = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_location_distance_from_business_district_03.hdf5')
# 是否容易寻找
location_easy_to_find = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_location_easy_to_find_03.hdf5')

# 服务
# 排队等候时间
service_wait_time = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_service_wait_time_05.hdf5')
# 服务人员态度
service_waiters_attitude = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_service_waiters_attitude_04.hdf5')
# 是否容易停车
service_parking_convenience = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_service_parking_convenience_04.hdf5')
# 点菜/上菜速度
service_serving_speed = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_service_serving_speed_05.hdf5')

# 价格
# 价格水平
price_level = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_price_level_05.hdf5')
# 性价比
price_cost_effective = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_price_cost_effective_04.hdf5')
# 折扣力度
price_discount = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_price_discount_05.hdf5')

# 环境
# 装修情况
environment_decoration = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_environment_decoration_05.hdf5')
# 嘈杂情况
environment_noise = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_environment_noise_03.hdf5')
# 就餐空间
environment_space = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_environment_space_05.hdf5')
# 卫生情况
environment_cleaness = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_environment_cleaness_03.hdf5')

# 菜品
# 分量
dish_portion = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_dish_portion_03.hdf5')
# 口感
dish_taste = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_dish_taste_05.hdf5')
# 外观
dish_look = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_dish_look_03.hdf5')
# 推荐程度
dish_recommendation = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_dish_recommendation_04.hdf5')

# 其他
# 本次消费感受
others_overall_experience = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_others_overall_experience_03.hdf5')
# 再次消费的意愿
others_willing_to_consume_again = os.path.join(os.path.abspath('./'), 'TextCNN_models_ckpt', 'textcnn_model_others_willing_to_consume_again_03.hdf5')


def load_tokener():
    with open(tokener_filepath, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def data_process(comment):
    comment = [' '.join(jieba.cut(comment))]
    tokenizer = load_tokener()
    query = tokenizer.texts_to_sequences(comment)
    query = sequence.pad_sequences(query, maxlen=maxlen)
    return query

def predict(query):
    # res: [
    # [category one， score，[[label1, sentiment1], [label2, sentiment2], [lebel, sentiment3], ..]]
    # [category two， score，[[label1, sentiment1], [label2, sentiment2], [lebel, sentiment3], ..]]
    # ...
    # ]
    # label: 0正面， 1中性，2反面， 3未提及
    query = data_process(query)
    res = []
    model = get_model()

    score = 0
    sentiment = []
    model.load_weights(location_traffic_convenience)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '交通便利'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '交通较便利'])
    elif sentiment_label == 2:
        score -= 1
        sentiment.append([sentiment_label, '交通不便'])
    else:
        sentiment.append([sentiment_label, None])
        # pass

    model.load_weights(location_distance_from_business_district)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '近商圈'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '距离商圈适中'])
    elif sentiment_label == 2:
        score -= 1
        sentiment.append([sentiment_label, '离商圈远'])
    else:
        sentiment.append([sentiment_label, None])
        pass

    model.load_weights(location_easy_to_find)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '位置显眼'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '位置较显眼'])
    elif sentiment_label == 2:
        score -= 1
        sentiment.append([sentiment_label, '位置隐蔽'])
    else:
        sentiment.append([sentiment_label, None])
        # pass
    res.append([0, score, sentiment])



    score = 0
    sentiment = []
    model.load_weights(service_wait_time)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '等位时间短'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '等位时间适中'])
    elif sentiment_label == 2:
        score -= 1
        sentiment.append([sentiment_label, '等位时间长'])
    else:
        sentiment.append([sentiment_label, None])
        # pass

    model.load_weights(service_waiters_attitude)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '服务非常好'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '服务可以'])
    elif sentiment_label == 2:
        score -= 1
        sentiment.append([sentiment_label, '服务差'])
    else:
        sentiment.append([sentiment_label, None])
        # pass

    model.load_weights(service_parking_convenience)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '停车很方便'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '较易停车'])
    elif sentiment_label == 2:
        score -= 1
        sentiment.append([sentiment_label, '车位紧张'])
    else:
        sentiment.append([sentiment_label, None])
        # pass

    model.load_weights(service_serving_speed)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '上菜快'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '上菜较快'])
    elif sentiment_label == 2:
        score -= 1
        sentiment.append([sentiment_label, '上菜慢'])
    else:
        sentiment.append([sentiment_label, None])
        # pass
    res.append([1, score, sentiment])



    score = 0
    sentiment = []
    model.load_weights(price_level)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '价格便宜'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '价格适中'])
    elif sentiment_label == 2:
        score -= 1
        sentiment.append([sentiment_label, '价格贵'])
    else:
        sentiment.append([sentiment_label, None])
        # pass

    model.load_weights(price_cost_effective)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '性价比高'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '性价比适中'])
    elif sentiment_label == 2:
        score -= 1
        sentiment.append([sentiment_label, '性价比低'])
    else:
        sentiment.append([sentiment_label, None])
        # pass

    model.load_weights(price_discount)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '折扣力度很大'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '有一定折扣力度'])
    elif sentiment_label == 2:
        score -= 1
        sentiment.append([sentiment_label, '折扣力度太小'])
    else:
        sentiment.append([sentiment_label, None])
        # pass
    res.append([2, score, sentiment])


    score = 0
    sentiment = []
    model.load_weights(environment_decoration)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '装修很好'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '装修可以'])
    elif sentiment_label == 2:
        score -= 1
        sentiment.append([sentiment_label, '装修差'])
    else:
        sentiment.append([sentiment_label, None])
        # pass

    model.load_weights(environment_noise)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '环境安静舒适'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '环境不算嘈杂'])
    elif sentiment_label == 2:
        score -= 1
        sentiment.append([sentiment_label, '环境嘈杂'])
    else:
        sentiment.append([sentiment_label, None])
        # pass

    model.load_weights(environment_space)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '空间很宽敞'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '空间较为宽敞'])
    elif sentiment_label == 2:
        score -= 1
        sentiment.append([sentiment_label, '空间狭小'])
    else:
        sentiment.append([sentiment_label, None])
        # pass

    model.load_weights(environment_cleaness)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '非常干净整洁'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '卫生情况可以'])
    elif sentiment_label == 2:
        score -= 1
        sentiment.append([sentiment_label, '卫生情况差'])
    else:
        sentiment.append([sentiment_label, None])
        # pass
    res.append([3, score, sentiment])



    score = 0
    sentiment = []
    model.load_weights(dish_portion)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '分量大'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '分量适中'])
    elif sentiment== 2:
        score -= 1
        sentiment.append([sentiment_label, '分量少'])
    else:
        sentiment.append([sentiment_label, None])
        # pass

    model.load_weights(dish_taste)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '口感很好'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '口感可以'])
    elif sentiment == 2:
        score -= 1
        sentiment.append([sentiment_label, '口感差'])
    else:
        sentiment.append([sentiment_label, None])
        # pass

    model.load_weights(dish_look)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '菜品精致'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '菜品观感可以'])
    elif sentiment == 2:
        score -= 1
        sentiment.append([sentiment_label, '菜品观感差'])
    else:
        sentiment.append([sentiment_label, None])
        # pass


    model.load_weights(dish_recommendation)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '强烈推荐'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '可以一试'])
    elif sentiment == 2:
        score -= 1
        sentiment.append([sentiment_label, '不推荐'])
    else:
        sentiment.append([sentiment_label, None])
        # pass
    res.append([4, score, sentiment])



    score = 0
    sentiment = []
    model.load_weights(others_overall_experience)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '体验极佳，点赞'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '感觉还可以'])
    elif sentiment == 2:
        score -= 1
        sentiment.append([sentiment_label, '体验极差'])
    else:
        sentiment.append([sentiment_label, None])
        # pass

    model.load_weights(others_willing_to_consume_again)
    sentiment_label = np.argmax(model.predict(query)[0])
    if sentiment_label == 0:
        score += 1
        sentiment.append([sentiment_label, '下次一定再来'])
    elif sentiment_label == 1:
        sentiment.append([sentiment_label, '下次看心情'])
    elif sentiment == 2:
        score -= 1
        sentiment.append([sentiment_label, '感觉不会再爱了'])
    else:
        sentiment.append([sentiment_label, None])
    res.append([5, score, sentiment])


    return res




if __name__ == '__main__':
    # res = predict('我想说他们家的优惠活动好持久啊，我预售的时候买的券，前两天心血来潮去吃的活动还在继续首先说下服务，因为和男票开车去的，有点不认路，老板很耐心的在电话里帮我们指路，到了门店之后也帮我们推荐了他们家做的比较地道的伤心凉粉，说是厨师是四川那边来的。环境呢比较简单干净，去的时候下午一点多了，还有四五桌人在用餐口味对于我而言点了麻辣的口感正正好，男票比较能吃辣，相对而言觉得他们家的麻辣口感麻有了，辣还欠缺一点，老板娘说考虑到客人口味不同所以没敢放太多辣椒，能吃辣的朋友可以考虑下单之前和老板先说好。鱼呢我们选的是黑鱼，2.9斤的鱼加上一盆我以为没有什么东西实际上东西很多的锅底，我们吃的饱饱的，最后以为吃的差不多了，打包一看简直像没动过一样，分量还是满足的，鱼比较新鲜。伤心凉粉很辣，不过口味也蛮好吃的。总的来说，性价比还是可以的，两个人吃了大概160左右，用了团购券的话一百块不到，会考虑下次再来')
    res = predict('运气很好，抽中了大众点评的霸王餐。这家主题餐厅心仪已久了，种种原因一直未能成行，没想到抽中了150元的代金券~~~Lucky!!!【位置】这家店位于星游城1楼，就在轨道交通4号线上，交通便利，但是停车场非常考验技术，转得头都晕了。。。【环境】主题餐厅啊，装修的非常好，大赞，处处都是海盗风格！非常喜欢！【服务】每次有人进来，服务员都会高呼“欢迎入伙~”。如果点了海盗的宝藏，他们也会高呼。我点了火烧鹦鹉，服务员还特意问“要不要拍个照啊~”哈哈~~【口味】总体来说还是不错，推荐火烧鹦鹉，很香！周围吃饭的小朋友很多，感觉这是家很适合朋友聚餐的店~~！')
    print(res)

