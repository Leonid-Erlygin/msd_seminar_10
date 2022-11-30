import torch

from hubconf import CPC_audio
cpc = CPC_audio(pretrained=True)


from cpc.eval.linear_separability import run, parse_args
from cpc.dataset import AudioBatchData, findAllSeqs, filterSeqs, parseSeqLabels
import cpc.criterion as cr



argv = [
 '/app/data/LibriSpeech',
 '/app/data/LibriSpeech/train_split_small.txt',
 '/app/data/LibriSpeech/test_split_small.txt',
 '/app/data/checkpoints/60k_epoch4-d0f474de.pt',
 '--pathCheckpoint',
 '/app/data/checkpoints/',
 '--pathPhone',
 '/app/data/converted_aligned_phones.txt']

args = parse_args(argv)



logs = {"epoch": [], "iter": [], "saveStep": -1}
load_criterion = False

seqNames, speakers = findAllSeqs(args.pathDB,
                                    extension=args.file_extension,
                                    loadCache=not args.ignore_cache)

model, hidden_gar, hidden_encoder = cpc, cpc.gAR.getDimOutput(), cpc.gEncoder.getDimOutput()

model.cuda()
model = torch.nn.DataParallel(model, device_ids=range(args.nGPU))

dim_features = hidden_encoder if args.get_encoded else hidden_gar

# Now the criterion
phone_labels = None

phone_labels, n_phones = parseSeqLabels(args.pathPhone)

print(f"Running phone separability with aligned phones")
criterion = cr.PhoneCriterion(dim_features,
                                n_phones, args.get_encoded)

criterion.cuda()
criterion = torch.nn.DataParallel(criterion, device_ids=range(args.nGPU))



# Dataset
seq_train = filterSeqs(args.pathTrain, seqNames)
seq_val = filterSeqs(args.pathVal, seqNames)


db_train = AudioBatchData(args.pathDB, args.size_window, seq_train,
                            phone_labels, len(speakers))
db_val = AudioBatchData(args.pathDB, args.size_window, seq_val,
                        phone_labels, len(speakers))

batch_size = args.batchSizeGPU * args.nGPU

train_loader = db_train.getDataLoader(batch_size, "uniform", True,
                                        numWorkers=1)

val_loader = db_val.getDataLoader(batch_size, 'sequential', False,
                                    numWorkers=1)




# Optimizer
g_params = list(criterion.parameters())
model.optimize = False
model.eval()
if args.unfrozen:
    print("Working in full fine-tune mode")
    g_params += list(model.parameters())
    model.optimize = True
else:
    print("Working with frozen features")
    for g in model.parameters():
        g.requires_grad = False

optimizer = torch.optim.Adam(g_params, lr=args.lr,
                                betas=(args.beta1, args.beta2),
                                eps=args.epsilon)


from pathlib import Path
import json
# Checkpoint directory
args.pathCheckpoint = Path(args.pathCheckpoint)
args.pathCheckpoint.mkdir(exist_ok=True)
args.pathCheckpoint = str(args.pathCheckpoint / "checkpoint")

with open(f"{args.pathCheckpoint}_args.json", 'w') as file:
    json.dump(vars(args), file, indent=2)

run(model, criterion, train_loader, val_loader, optimizer, logs,
    args.n_epoch, args.pathCheckpoint)