import torch_geometric
import pandas as pd
import torch
import dgl
import statistics
import networkx as nx
import numpy as np
import tqdm
from dataloader import NodeNegativeLoader


def get_info(df, txt):
    info = {}
    for index, row in df.iterrows():
        idx = row['graph']['node_idx']
        context = row['conversations'][0]['value']
        f_label = row['conversations'][1]['value']
        f_abs = context.split('Abstract: ')[-1].split(' \n Question:')[0]

        title, abs, label = txt[idx][1], txt[idx][2], txt[idx][3].strip()
        assert label == f_label, f"{label}, {f_label}"
        assert f_abs.startswith(title)
        assert label != 'nan'

        info[idx] = [title, abs, label]
    return info



if __name__ == '__main__':
    test_info = pd.read_json('./cora/cora_test_instruct_std.json')
    txt = []
    with open('./cora/train_text.txt') as f:
        for i, line in enumerate(f.readlines()):
            txt.append(line.split('\t'))
    
    info = get_info(test_info, txt)
    print(len(txt))
    data = torch.load('./graph_data_paper.pt')['cora']
    print(data)
    print(data.edge_index.shape)
    data = torch_geometric.transforms.ToUndirected()(data)
    data = torch_geometric.transforms.AddRemainingSelfLoops()(data)
    print(data.edge_index.shape)
    # val_id = torch.nonzero(data.val_mask, as_tuple=False).squeeze()
    test_nid = torch.nonzero(data.test_mask, as_tuple=False).squeeze()
    assert len(info.keys()) == test_nid.shape[0], f"info length: {len(info.keys())}, test id: {test_nid.shape[0]}"

    nid = torch.LongTensor(list(info.keys()))
    print(test_nid, nid)

    prompts = []
    outputs = []
    edge_indexs = []
    node_set = []
    X = []
    cnt = 0
    s = nid.shape[0]
    # print(train_idx)
    for i in tqdm.tqdm(range(nid.shape[0])):
        node = nid[i].reshape(-1)
        neighbor_sampler = NodeNegativeLoader(data, num_neighbors=[25, 10], neg_ratio=0., batch_size=1, shuffle=False, input_nodes=node)
        sub_g = next(iter(neighbor_sampler))[0]
        assert sub_g.n_id[0] == node

        title, abs, field = info[node.data.item()]
        break

        if sub_g.edge_index.shape[1] < 1:  # 数据集存在孤立节点xw
            cnt += 1

        # prompt = f"""Given a representation of a paper: <Node 1>, with the following information: \nAbstract: {abs} \n Question: Which of the following subcategories of computer science does this paper belong to: 1. artificial intelligence, agents 2. artificial intelligence, data mining 3. artificial intelligence, expert systems 4. artificial intelligence, games and search 5. artificial intelligence, knowledge representation 6. artificial intelligence, machine learning, case-based 7. artificial intelligence, machine learning, genetic algorithms 8. artificial intelligence, machine learning, neural networks 9. artificial intelligence, machine learning, probabilistic methods 10. artificial intelligence, machine learning, reinforcement learning 11. artificial intelligence, machine learning, rule learning 12. artificial intelligence, machine learning, theory 13. artificial intelligence, nlp 14. artificial intelligence, planning 15. artificial intelligence, robotics 16. artificial intelligence, speech 17. artificial intelligence, theorem proving 18. artificial intelligence, vision and pattern recognition 19. data structures  algorithms and theory, computational complexity 20. data structures  algorithms and theory, computational geometry 21. data structures  algorithms and theory, formal languages 22. data structures  algorithms and theory, hashing 23. data structures  algorithms and theory, logic 24. data structures  algorithms and theory, parallel 25. data structures  algorithms and theory, quantum computing 26. data structures  algorithms and theory, randomized 27. data structures  algorithms and theory, sorting 28. databases, concurrency 29. databases, deductive 30. databases, object oriented 31. databases, performance 32. databases, query evaluation 33. databases, relational 34. databases, temporal 35. encryption and compression, compression 36. encryption and compression, encryption 37. encryption and compression, security 38. hardware and architecture, distributed architectures 39. hardware and architecture, high performance computing 40. hardware and architecture, input output and storage 41. hardware and architecture, logic design 42. hardware and architecture, memory structures 43. hardware and architecture, microprogramming 44. hardware and architecture, vlsi 45. human computer interaction, cooperative 46. human computer interaction, graphics and virtual reality 47. human computer interaction, interface design 48. human computer interaction, multimedia 49. human computer interaction, wearable computers 50. information retrieval, digital library 51. information retrieval, extraction 52. information retrieval, filtering 53. information retrieval, retrieval 54. nan 55. networking, internet 56. networking, protocols 57. networking, routing 58. networking, wireless 59. operating systems, distributed 60. operating systems, fault tolerance 61. operating systems, memory management 62. operating systems, realtime 63. programming, compiler design 64. programming, debugging 65. programming, functional 66. programming, garbage collection 67. programming, java 68. programming, logic 69. programming, object oriented 70. programming, semantics 71. programming, software development ? Directly give the full name of the most likely category of this paper."""
        # prompt = f"""Given a representation of a paper: <Node 1>, with the following information: \nAbstract: {abs} \n Title: {title} \n Question: Which of the following subcategories of computer science does this paper belong to: 1. artificial intelligence, agents 2. artificial intelligence, data mining 3. "artificial intelligence, expert systems" 4. "artificial intelligence, games and search 5. artificial intelligence, knowledge representation 6. artificial intelligence, machine learning, case-based 7. artificial intelligence, machine learning, genetic algorithms 8. artificial intelligence, machine learning, neural networks 9. artificial intelligence, machine learning, probabilistic methods 10. artificial intelligence, machine learning, reinforcement learning 11. artificial intelligence, machine learning, rule learning 12. artificial intelligence, machine learning, theory 13. artificial intelligence, nlp 14. artificial intelligence, planning 15. artificial intelligence, robotics 16. artificial intelligence, speech 17. artificial intelligence, theorem proving 18. artificial intelligence, vision and pattern recognition 19. data structures  algorithms and theory, computational complexity 20. data structures  algorithms and theory, computational geometry 21. data structures  algorithms and theory, formal languages 22. data structures  algorithms and theory, hashing 23. data structures  algorithms and theory, logic 24. data structures  algorithms and theory, parallel 25. data structures  algorithms and theory, quantum computing 26. data structures  algorithms and theory, randomized 27. data structures  algorithms and theory, sorting 28. databases, concurrency 29. databases, deductive 30. databases, object oriented 31. databases, performance 32. databases, query evaluation 33. databases, relational 34. databases, temporal 35. encryption and compression, compression 36. encryption and compression, encryption 37. encryption and compression, security 38. hardware and architecture, distributed architectures 39. hardware and architecture, high performance computing 40. hardware and architecture, input output and storage 41. hardware and architecture, logic design 42. hardware and architecture, memory structures 43. hardware and architecture, microprogramming 44. hardware and architecture, vlsi 45. human computer interaction, cooperative 46. human computer interaction, graphics and virtual reality 47. human computer interaction, interface design 48. human computer interaction, multimedia 49. human computer interaction, wearable computers 50. information retrieval, digital library 51. information retrieval, extraction 52. information retrieval, filtering 53. information retrieval, retrieval 54. nan 55. networking, internet 56. networking, protocols 57. networking, routing 58. networking, wireless 59. operating systems, distributed 60. operating systems, fault tolerance 61. operating systems, memory management 62. operating systems, realtime 63. programming, compiler design 64. programming, debugging 65. programming, functional 66. programming, garbage collection 67. programming, java 68. programming, logic 69. programming, object oriented 70. programming, semantics 71. programming, software development ? Directly give the most likely category of this paper from these categories."""
        prompt = f"""Given a representation of a paper: <Node 1>, with the following information: \nAbstract: {abs} \n Title: {title} \n Question: Which subcategory does this paper belong to? Please directly give the most likely answer from the following subcategories: "artificial intelligence, agents", "artificial intelligence, data mining", "artificial intelligence, expert systems", "artificial intelligence, games and search", "artificial intelligence, knowledge representation", "machine learning, case-based", "machine learning, genetic algorithms", "machine learning, neural networks", "machine learning, probabilistic methods", "machine learning, reinforcement learning", "machine learning, rule learning", "machine learning, theory", "artificial intelligence, nlp", "artificial intelligence, planning", "artificial intelligence, robotics", "artificial intelligence, speech", "artificial intelligence, theorem proving", "artificial intelligence, vision and pattern recognition", "data structures  algorithms and theory, computational complexity", "data structures  algorithms and theory, computational geometry", "data structures  algorithms and theory, formal languages", "data structures  algorithms and theory, hashing", "data structures  algorithms and theory, logic", "data structures  algorithms and theory, parallel", "data structures  algorithms and theory, quantum computing", "data structures  algorithms and theory, randomized", "data structures  algorithms and theory, sorting", "databases, concurrency", "databases, deductive", "databases, object oriented", "databases, performance", "databases, query evaluation", "databases, relational", "databases, temporal", "encryption and compression, compression", "encryption and compression, encryption", "encryption and compression, security", "hardware and architecture, distributed architectures", "hardware and architecture, high performance computing", "hardware and architecture, input output and storage", "hardware and architecture, logic design", "hardware and architecture, memory structures", "hardware and architecture, microprogramming", "hardware and architecture, vlsi", "human computer interaction, cooperative", "human computer interaction, graphics and virtual reality", "human computer interaction, interface design", "human computer interaction, multimedia", "human computer interaction, wearable computers", "information retrieval, digital library", "information retrieval, extraction", "information retrieval, filtering", "information retrieval, retrieval", "nan", "networking, internet", "networking, protocols", "networking, routing", "networking, wireless", "operating systems, distributed", "operating systems, fault tolerance", "operating systems, memory management", "operating systems, realtime", "programming, compiler design", "programming, debugging", "programming, functional", "programming, garbage collection", "programming, java", "programming, logic", "programming, object oriented", "programming, semantics" or "programming, software development"."""
        # prompt = f"""Given a representation of a paper: <Node 1>, with the following information: \nAbstract: {abs} \n Question: Which of the following subcategories of computer science does this paper belong to: 1. artificial intelligence, agents 2. artificial intelligence, data mining 3. artificial intelligence, expert systems 4. artificial intelligence, games and search 5. artificial intelligence, knowledge representation 6. artificial intelligence, machine learning, case-based 7. artificial intelligence, machine learning, genetic algorithms 8. artificial intelligence, machine learning, neural networks 9. artificial intelligence, machine learning, probabilistic methods 10. artificial intelligence, machine learning, reinforcement learning 11. artificial intelligence, machine learning, rule learning 12. artificial intelligence, machine learning, theory 13. artificial intelligence, nlp 14. artificial intelligence, planning 15. artificial intelligence, robotics 16. artificial intelligence, speech 17. artificial intelligence, theorem proving 18. artificial intelligence, vision and pattern recognition 19. data structures  algorithms and theory, computational complexity 20. data structures  algorithms and theory, computational geometry 21. data structures  algorithms and theory, formal languages 22. data structures  algorithms and theory, hashing 23. data structures  algorithms and theory, logic 24. data structures  algorithms and theory, parallel 25. data structures  algorithms and theory, quantum computing 26. data structures  algorithms and theory, randomized 27. data structures  algorithms and theory, sorting 28. databases, concurrency 29. databases, deductive 30. databases, object oriented 31. databases, performance 32. databases, query evaluation 33. databases, relational 34. databases, temporal 35. encryption and compression, compression 36. encryption and compression, encryption 37. encryption and compression, security 38. hardware and architecture, distributed architectures 39. hardware and architecture, high performance computing 40. hardware and architecture, input output and storage 41. hardware and architecture, logic design 42. hardware and architecture, memory structures 43. hardware and architecture, microprogramming 44. hardware and architecture, vlsi 45. human computer interaction, cooperative 46. human computer interaction, graphics and virtual reality 47. human computer interaction, interface design 48. human computer interaction, multimedia 49. human computer interaction, wearable computers 50. information retrieval, digital library 51. information retrieval, extraction 52. information retrieval, filtering 53. information retrieval, retrieval 54. nan 55. networking, internet 56. networking, protocols 57. networking, routing 58. networking, wireless 59. operating systems, distributed 60. operating systems, fault tolerance 61. operating systems, memory management 62. operating systems, realtime 63. programming, compiler design 64. programming, debugging 65. programming, functional 66. programming, garbage collection 67. programming, java 68. programming, logic 69. programming, object oriented 70. programming, semantics 71. programming, software development ? Please give one answer of these subcategories directly."""

        outputs.append(field)
        prompts.append(prompt)
        edge_indexs.append(sub_g.edge_index.data.tolist())
        node_set.append(sub_g.n_id.data.tolist())
        X.append(sub_g.x.data.tolist())


# print(f"{cnt} / {s}")
# df = pd.DataFrame({
#     "prompt": prompts,
#     "output": outputs,
#     "edge_index": edge_indexs,
#     "node_set":node_set,
#     "x": X,
# })
# df.to_json("../instruction/cora/cora_dataset_test.json", force_ascii=False)
