import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import time
from learnable_diffs import load_compatible_model
from my_funcs import *
import torchvision.utils as vutils
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])


def save_captions(args, word_map, hypotheses, references):
    result_json_file = {}
    reference_json_file = {}
    kkk = -1
    for item in hypotheses:
        kkk += 1
        line_hypo = ""

        for word_idx in item:
            word = get_key(word_map, word_idx)
            # print(word)
            line_hypo += word[0] + " "

        result_json_file[str(kkk)] = []
        result_json_file[str(kkk)].append(line_hypo)

        line_hypo += "\r\n"

    kkk = -1
    for item in tqdm(references):
        kkk += 1

        reference_json_file[str(kkk)] = []

        for sentence in item:
            line_repo = ""
            for word_idx in sentence:
                word = get_key(word_map, word_idx)
                line_repo += word[0] + " "
            reference_json_file[str(kkk)].append(line_repo)
            line_repo += "\r\n"

    with open('eval_results_fortest/' + args.Split +'/'+args.encoder_image + "_" +args.encoder_feat+"_" +args.decoder + '_res.json', 'w') as f:
        json.dump(result_json_file, f)

    with open('eval_results_fortest/' + args.Split +'/'+ args.encoder_image + "_" +args.encoder_feat+"_" +args.decoder + '_gts.json', 'w') as f:
        json.dump(reference_json_file, f)

def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value]

def evaluate_transformer(args, encoder_image, chg_filter, encoder_feat, decoder, device, my_log):
    # Load model
    encoder_image = encoder_image.to(device)
    encoder_image.eval()
    encoder_image.device = device
    chg_filter = chg_filter.to(device)
    chg_filter.eval()
    encoder_feat = encoder_feat.to(device)
    encoder_feat.eval()
    decoder = decoder.to(device)
    decoder.eval()
    decoder.device = device

    # Load word map (word2ix)
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as f:
        word_map = json.load(f)

    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    """
    Evaluation for decoder: transformer
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    beam_size = args.beam_size
    Caption_End = False
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, args.Split, transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    references = list()
    hypotheses = list()
    change_references = list()
    change_hypotheses = list()
    nochange_references = list()
    nochange_hypotheses = list()
    change_acc=0
    nochange_acc=0

    with torch.no_grad():
        for i, (image_pairs, caps, caplens, allcaps) in enumerate(
                tqdm(loader, desc=args.Split + " EVALUATING AT BEAM SIZE " + str(beam_size))):
            # 5 image is the same when "shuffle=False" of the dataloader
            if (i + 1) % 5 != 0:
                continue

            k = beam_size

            # Move to GPU device, if available
            image_pairs = image_pairs.to(device)  # [1, 2, 3, 256, 256]

            # Encode
            imgs_A = image_pairs[:, 0, :, :, :]
            imgs_B = image_pairs[:, 1, :, :, :]


            imgs_A = encoder_image(imgs_A)
            imgs_B = encoder_image(imgs_B)  # encoder_image :[1, 1024,14,14]

            if args.encoder == 'resnet':
                batch_size = imgs_A.shape[0]
                h = imgs_A.shape[2]
                w = imgs_A.shape[3]
                node_num = h * w
                device = imgs_A.device
            elif args.encoder == 'transformer':
                raise NotImplementedError
            else:
                raise NotImplementedError

            full_edge = get_full_edge(batch_size, node_num, device)
            imgA_graph, self_adjA = chg_filter(imgs_A, full_edge)
            imgB_graph, self_adjB = chg_filter(imgs_B, full_edge)

            # [batch, token]
            adj_diff = get_filter_adj(self_adjA, self_adjB)
            # [batch, token]
            adj_diff = get_soft_topk(adj_diff, top_k=args.top_k)
            # [batch, 8, token]
            adj_diff = adj_diff.view(batch_size, args.mask_head, node_num).repeat_interleave(int(args.graph_out_d / args.mask_head), dim=1)
            # [batch, 1, h, w]
            adj_diff = adj_diff.view(batch_size, args.graph_out_d, h, w).contiguous()

            mask = adj_diff
            imgA_graph = imgA_graph.view(batch_size, h, w, -1).transpose(2, 3).transpose(1, 2).contiguous()
            imgB_graph = imgB_graph.view(batch_size, h, w, -1).transpose(2, 3).transpose(1, 2).contiguous()
            img_A_fil = imgA_graph * mask
            img_B_fil = imgB_graph * mask

            fused_feat = encoder_feat(batch_size, img_A_fil, img_B_fil)
            fused_feat = fused_feat.transpose(0, 1)  # [h*w, batch_size, 2*dim]

            tgt = torch.zeros(52, k).to(device).to(torch.int64)
            tgt_length = tgt.size(0)
            mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            mask = mask.to(device)

            tgt[0, :] = torch.LongTensor([word_map['<start>']]*k).to(device) # k_prev_words:[52,k]
            # Tensor to store top k sequences; now they're just <start>
            seqs = torch.LongTensor([[word_map['<start>']]*1] * k).to(device)  # [1,k]
            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(device)
            # Lists to store completed sequences and scores
            complete_seqs = []
            complete_seqs_scores = []
            step = 1

            k_prev_words = tgt.permute(1,0)
            S = fused_feat.size(0)
            encoder_dim = fused_feat.size(-1)

            # # We'll treat the problem as having a batch size of k, where k is beam_size
            fused_feat = fused_feat.expand(S,k, encoder_dim)  # [S,k, encoder_dim]
            encoder_out = fused_feat.permute(1,0,2)

            # Start decoding
            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                tgt = k_prev_words.permute(1,0)
                tgt_embedding = decoder.vocab_embedding(tgt)
                tgt_embedding = decoder.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

                encoder_out = encoder_out.permute(1, 0, 2)
                pred = decoder.transformer(tgt_embedding, encoder_out, tgt_mask=mask)  # (length, batch, feature_dim)
                encoder_out = encoder_out.permute(1, 0, 2)
                pred = decoder.wdc(pred)  # (length, batch, vocab_size)
                scores = pred.permute(1,0,2)  # (batch,length,  vocab_size)
                scores = scores[:, step - 1, :].squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]
                scores = F.log_softmax(scores, dim=1)
                # top_k_scores: [s, 1]
                scores = top_k_scores.expand_as(scores) + scores  # [s, vocab_size]
                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='floor')
                next_word_inds = top_k_words % vocab_size

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                # Set aside complete sequences
                if len(complete_inds) > 0:
                    Caption_End = True
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly
                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                # Important: this will not work, since decoder has self-attention
                # k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1).repeat(k, 52)
                k_prev_words = k_prev_words[incomplete_inds]
                k_prev_words[:, :step + 1] = seqs  # [s, 52]
                # k_prev_words[:, step] = next_word_inds[incomplete_inds]  # [s, 52]
                # Break if things have been going on too long
                if step > 50:
                    break
                step += 1


            # choose the caption which has the best_score.
            if (len(complete_seqs_scores) == 0):
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            if (len(complete_seqs_scores) > 0):
                assert Caption_End
                indices = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[indices]
                # References
                img_caps = allcaps[0].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads

                references.append(img_captions)
                # Hypotheses
                new_sent = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
                hypotheses.append(new_sent)
                assert len(references) == len(hypotheses)

                nochange_list = ["the scene is the same as before ", "there is no difference ",
                                 "the two scenes seem identical ", "no change has occurred ",
                                 "almost nothing has changed "]
                ref_sentence = img_captions[1]
                ref_line_repo = ""
                for ref_word_idx in ref_sentence:
                    ref_word = get_key(word_map, ref_word_idx)
                    ref_line_repo += ref_word[0] + " "

                hyp_sentence = new_sent
                hyp_line_repo = ""
                for hyp_word_idx in hyp_sentence:
                    hyp_word = get_key(word_map, hyp_word_idx)
                    hyp_line_repo += hyp_word[0] + " "
                if ref_line_repo not in nochange_list:
                    change_references.append(img_captions)
                    change_hypotheses.append(new_sent)
                    if hyp_line_repo not in nochange_list:
                        change_acc = change_acc+1
                else:
                    nochange_references.append(img_captions)
                    nochange_hypotheses.append(new_sent)
                    if hyp_line_repo in nochange_list:
                        nochange_acc = nochange_acc+1


    print('len(nochange_references):', len(nochange_references))
    print('len(change_references):', len(change_references))
    # Calculate BLEU1~4, METEOR, ROUGE_L, CIDEr scores
    if len(nochange_references)>0:
        print('nochange_metric:')
        nochange_metric = get_eval_score(nochange_references, nochange_hypotheses)
        print("nochange_acc:", nochange_acc / len(nochange_references))
    if len(change_references)>0:
        print('change_metric:')
        change_metric = get_eval_score(change_references, change_hypotheses)
        print("change_acc:", change_acc / len(change_references))
    print(".......................................................")
    metrics = get_eval_score(references, hypotheses)
    print_log("{} - beam size {}: BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
              (args.decoder, args.beam_size, metrics["Bleu_1"], metrics["Bleu_2"], metrics["Bleu_3"],
               metrics["Bleu_4"],
               metrics["METEOR"], metrics["ROUGE_L"], metrics["CIDEr"]), my_log)


    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change_Captioning')
    parser.add_argument('--data_folder', default="./data",help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default="LEVIR_CC_5_cap_per_img_5_min_word_freq",help='base name shared by data files.')

    parser.add_argument('--encoder_image', default="resnet101")
    parser.add_argument('--encoder_feat', default="filter_KV")
    parser.add_argument('--decoder', default="trans", help="decoder img2txt")  #
    parser.add_argument('--Split', default="TEST", help='which')
    parser.add_argument('--epoch', default="epoch", help='which')
    parser.add_argument('--beam_size', type=int, default=1, help='beam_size.')

    parser.add_argument('--path', default="./models_checkpoint/", help='model checkpoint.')
    parser.add_argument('--gpu', type=int, default=1)

    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--savedir', default="./models_checkpoint/")
    parser.add_argument('--tag', type=str, default="LEVIR_20251231")
    parser.add_argument('--graph_out_d', type=int, default=1024, help='output feature dim of graph')

    parser.add_argument('--encoder', type=str, default='resnet')
    parser.add_argument('--top_k', type=int, default=128, help='num of filter feature')
    parser.add_argument('--sig_w', type=float, default=50.0, help='scale factor for sigmoid')
    parser.add_argument('--kv_new', type=bool, default=True, help='kv update during fusion attention')
    parser.add_argument('--mask_head', type=int, default=8, help='multi-head num of mask')
    parser.add_argument('--eval_old', type=bool, default=False, help='eval with old version ckpts')
    args = parser.parse_args()

    ckpt_path = os.path.join(args.savedir, args.tag)
    log_name = os.path.join(ckpt_path, 'test_log.txt')
    test_log = open(log_name, 'a+')

    filename = os.listdir(ckpt_path)
    for i in range(len(filename)):
        if '.txt' in filename[i]:
            continue

        print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

        checkpoint_path = os.path.join(ckpt_path, filename[i])
        print(args.path + filename[i])

        device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')

        # Load model
        if args.eval_old is True:
            checkpoint = load_compatible_model(checkpoint_path, device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=str(device))
        encoder_image = checkpoint['encoder_image']
        chg_filter = checkpoint['chg_filter']
        encoder_feat = checkpoint['encoder_feat']
        decoder = checkpoint['decoder']


        if args.decoder == "trans":
            metrics = evaluate_transformer(args, encoder_image, chg_filter, encoder_feat, decoder, device, test_log)

        print("{} - beam size {}: BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
              (args.decoder, args.beam_size, metrics["Bleu_1"], metrics["Bleu_2"], metrics["Bleu_3"],
               metrics["Bleu_4"],
               metrics["METEOR"], metrics["ROUGE_L"], metrics["CIDEr"]))
        print("\n")
        print("\n")

        time.sleep(10)
