import sys, os
from IPython import embed
import numpy as np
import pandas as pd
import argparse
import json
from tqdm import tqdm
import random
from sklearn.metrics import classification_report, accuracy_score
import time

np.random.seed(345)
random.seed(345)

import openai
from apiKeys import openai_api_key
openai.api_key = openai_api_key

def loadSIQA(args):
  trn = pd.read_csv("siqa.trnOnlyWDims.csv")
  if args.debug_trn:
    trn = trn.sample(args.debug_trn)  
  
  dev = pd.read_json("socialIWa_v1.4_dev_wDims.jsonl",lines=True)
  #dev = pd.read_json("socialIWa_v1.4_tst_wDims.jsonl",lines=True)
  if args.debug_dev:
    dev = dev.sample(args.debug_dev)

  for df in [ trn, dev]:
    df["answer"] = df[["label_ix","answerA","answerB","answerC"]].apply(
      lambda x: x[1:][x[0]],axis=1)
  

  return trn, dev, ["context","question","answer","answerA","answerB","answerC"]

def loadTOMI(args):
  trn = pd.read_csv("ToMi.trn.csv")
  if args.debug_trn:
    trn = trn.sample(args.debug_trn)

  trn["cands"] = trn[["answerMem","answerReal"]].apply(list,axis=1)

  dev = pd.read_csv("ToMi-finalNeuralTOM.csv")
  if args.debug_dev:
    dev = dev.sample(args.debug_dev)

  dev["cands"] = dev["cands"].apply(json.loads)
  for df in [dev, trn]:
    df["cands"].apply(random.shuffle)
    
    df["answerA"] = df["cands"].apply(lambda x: x[0])
    df["answerB"] = df["cands"].apply(lambda x: x[1])
    if args.instructions:
      df["story"] = df["story"].apply(lambda x: args.instructions+x)
  return trn, dev, ["story","question","answer","answerA","answerB"]

##############################################################
def _sampleExamples(t,n=3,trn=None,typeCol=None,stratCol=None):
  if n==0:
    return pd.DataFrame(data=[])
  if typeCol:
    trn = trn[trn[typeCol] == t]
    
  if stratCol:
    exs = []
    nUniqueTypes = trn[stratCol].drop_duplicates().shape[0]
    N = int(np.ceil(n/nUniqueTypes))
    for ix, c in trn.groupby(stratCol):
      exs.append(c.sample(N))
  
    out = pd.concat(exs).sample(n)
  else:
    out = trn.sample(n)
  return out

def sampleExamples(dev,trn,args):
  stratCol = args.stratify_examples_by
  typeCol = args.examples_of_same_type
  iterCol = typeCol if typeCol else "question"

  tqdm.pandas(ascii=True,desc="Getting training examples")
  dev["trnExamples"] = dev[iterCol].progress_apply(_sampleExamples,n=args.n_examples,trn=trn,
                                                typeCol=typeCol,stratCol=stratCol)
  return dev

##############################################################
def _formatExample(r,cols,probing_type,onlyTrueAnswer=False):
  contextCol, questionCol, answerCol = cols[:3]
  candCols = cols[3:]
  if probing_type == "lm":
    pref = r[contextCol] + " " + r[questionCol]
    out = [pref+" "+r[c] for c in candCols]

    if onlyTrueAnswer:
      out = out[[r[c] for c in candCols].index(r[answerCol])]
      
  elif probing_type == "mc":
    # random.shuffle(candCols)
    pref = r[contextCol] + " " + r[questionCol]
    out = pref + "\n" + "\n".join([l+": "+r[a] for l,a in zip("ABCDEF",candCols)])
    out += "\nAnswer:"
    if onlyTrueAnswer:
      m = "ABCDEF"[[r[c] for c in candCols].index(r[answerCol])]
      out += " "+m

  elif probing_type == "nli":
    # Warning this part is not complete
    pref = r[contextCol] + " " + r[questionCol]
    out = pref + "\n\n" + "\n".join([l+": "+r[a] for l,a in zip("ABCDEF",candCols)])
    out += "\nAnswer:"
    if onlyTrueAnswer:
      m = "ABCDEF"[[r[c] for c in candCols].index(r[answerCol])]
      out += " "+m
  
  elif probing_type == "post_nli":
    pref = r[contextCol] + " " + r[questionCol]
    pref += "\nAnswer:"
    pref += r["answer_for_nli"]
    pref += "\n\n"
    pref += "Based on your answer, select the answers listed below:"
    out = pref + "\n" + "\n".join([l+": "+r[a] for l,a in zip("ABCDEF",candCols)])
    out += "\nPlease select one most likely answer, and reply only with A, B, or C"
    out += "\nAnswer:"
    if onlyTrueAnswer:
      m = "ABCDEF"[[r[c] for c in candCols].index(r[answerCol])]
      out += " "+m

  else:
    raise ValueError("Incorrect --probing_type, should be 'lm' or 'mc'")
    
  # out is a string, unless probing_type=lm and onlyTrueAnswer=False,
  # in which case it's a list of strings
  return out

def formatExamples(x,cols=None,probing_type=None):
  f = x.apply(_formatExample,cols=cols,probing_type=args.probing_type,
              onlyTrueAnswer=True,axis=1)
  out = "\n\n".join(f)
  return out

def combineExamples(x):
  trn, dev = x.values

  if isinstance(dev, list):
    out = [trn + "\n\n" + d for d in dev]
  else:
    out = trn + "\n\n" + dev

  return out

#######################################################################
def getGPTanswers(text,variant="ada",attempt=0):
  # Only chat mode so far
  time.sleep(0.5)
  text = text.strip()
  
  try:
    if "turbo" in variant or "gpt-4" in variant:
      turns = text.split("\n\n")
      #turnsWithRoles = [{"role":sp,"content":msg} for t in turns for sp,msg in zip(["user","assistant"],t.split("\n")[0])]
      #turnsWithRoles = [d for d in turnsWithRoles if d["content"] != ""]
      turnsWithRoles = [{"role":"user","content":(turns[0])}]
      r = openai.ChatCompletion.create(
        model=variant,
        messages=turnsWithRoles,
        # echo=mcProbing == 0, # only echo in LM-probing style
        temperature=0,
        max_tokens=100,
        # logprobs=mcProbing,
      )
      return r["choices"][0]["message"]["content"].strip()
  except (openai.error.APIError, openai.error.RateLimitError) as e:
    print(e)
    print("Sleeping for 10 seconds, attempt nb", attempt)
    time.sleep(10)
    if attempt>10:
      print("Reached attempt limit, giving up")
      return None
    else:
      print("Trying again")
      return getGPTanswers(text,variant=variant,attempt=attempt+1)
      
#######################################################################

def getGPT3prob(text,mcProbing=0,variant="ada",attempt=0,useChatTurnsAndRoles=True):
  time.sleep(0.5)
  assert "turbo" not in variant or mcProbing > 0, variant+" model does not work with LM-probing"

  text = text.strip()
  
  try:
    if "turbo" in variant or "gpt-4" in variant:
      if useChatTurnsAndRoles:
        # TODO: turn MC QA into question / answer ??
        turns = text.split("\n\n")
        turnsWithRoles = [{"role":sp,"content":msg} for t in turns for sp,msg in zip(["user","assistant"],t.split("\nAnswer:"))]
        turnsWithRoles = [d for d in turnsWithRoles if d["content"] != ""]
      else:
        turnsWithRoles = [{"role": "user", "content": text}]
      r = openai.ChatCompletion.create(
        model=variant,
        messages=turnsWithRoles,
        # echo=mcProbing == 0, # only echo in LM-probing style
        temperature=0,
        max_tokens=1, # why 2 here?
        # logprobs=mcProbing,
      )
      answer = r["choices"][0]["message"]["content"].strip()
      # Trim irrelevant characters
      answer = answer.replace(":","")
      if answer not in ["A", "B", "C"]:
        print(answer)
        answer=""
      # print(text,r)

      r["choices"][0]["logprobs"] = {"top_logprobs":[{answer: 0}]}
    else:
      r = openai.Completion.create(
        model=variant,
        prompt=text,
        echo=mcProbing == 0, # only echo in LM-probing style
        temperature=0,
        max_tokens=1,
        logprobs=mcProbing,
      )
  except (openai.error.APIError, openai.error.RateLimitError) as e:
    print(e)
    print("Sleeping for 10 seconds, attempt nb", attempt)
    time.sleep(10)
    if attempt>10:
      print("Reached attempt limit, giving up")
      return None
    else:
      print("Trying again")
      return getGPT3prob(text,mcProbing=mcProbing,variant=variant,attempt=attempt+1)
    
  # embed();exit()
  
  if mcProbing:
    logprobs = {k: v for k,v in r["choices"][0]["logprobs"]["top_logprobs"][0].items()}
    logprobs = {k.strip(): v for k,v in logprobs.items()}
    return pd.Series(logprobs)
  else:
    logprobs = r["choices"][0]["logprobs"]["token_logprobs"]
    # removing the first token prob cause it's null
    logprobs = logprobs[1:]
    # print(logprobs)

    return np.sum(logprobs)


def _LMProbeGPT3(exTup,variant="ada"):
  predsL = [getGPT3prob(c,variant=variant) for c in exTup]
  out = pd.Series(dict(zip("ABCDEF",predsL)))
  return out 
  
def LMProbeGPT3(exs,variant="ada"):
  tqdm.pandas(desc=f"Getting GPT3 ({variant}) preds",ascii=True)
  preds = exs.progress_apply(_LMProbeGPT3,variant=variant)
  return preds

def MCProbeGPT3(exs,nOptions=3,variant="ada"):
  tqdm.pandas(desc=f"Getting GPT3 ({variant}) preds",ascii=True)
  preds = exs.progress_apply(getGPT3prob,mcProbing=nOptions,variant=variant)
  return preds

def NLIgenerate(exs,variant="ada"):
  tqdm.pandas(desc=f"Getting GPT3 ({variant}) preds",ascii=True)
  preds = exs.progress_apply(getGPTanswers,variant=variant)
  return preds

def mapPred(x,answerMap):
  if x["pred"] not in answerMap:
    return ""
  
  return x[answerMap[x["pred"]]]
  
def computeAccuracies(df,args):
  overall_acc = (df["answer"] == df["predAnswer"]).mean()
  
  accs = {"overall": overall_acc}
  print("Overall accuracy",overall_acc)
  if args.group_accuracy_by:
    for col in args.group_accuracy_by:
      for g, dfChunk in df.groupby(col):
        accs[ col+"_"+str(g)] =  (dfChunk["answer"] == dfChunk["predAnswer"]).mean()

  accs = pd.Series(accs)
  print(accs)
  return accs
  

##############################################################
def main(args):

  if args.input_prediction_file:
    df = pd.read_pickle(args.input_prediction_file)
    accs = computeAccuracies(df,args)
    
    return 
    
  
  if args.task == "siqa":
    trn, dev, cols = loadSIQA(args)
  elif args.task == "tomi":
    trn, dev, cols = loadTOMI(args)

  contextCol, questionCol, answerCol = cols[:3]
  candCols = cols[3:]
  
  # sample examples
  devPrepped = sampleExamples(dev,trn,args)
  # devPrepped["formattedTrnExamples"] = devPrepped["trnExamples"].

  devPrepped["formattedTrnExamples"] = devPrepped["trnExamples"].apply(
    formatExamples,cols=cols,probing_type=args.probing_type)
  
  # prep examples
  devPrepped["formattedDevExamples"] = devPrepped.apply(
    _formatExample,cols=cols,probing_type=args.probing_type,axis=1)

  devPrepped["formattedFullString"] = devPrepped[["formattedTrnExamples","formattedDevExamples"]].apply(combineExamples,axis=1)

  if args.probing_type=="lm":
    preds = LMProbeGPT3(devPrepped["formattedFullString"],variant=args.model_variant)
  elif args.probing_type=="mc":
    preds = MCProbeGPT3(devPrepped["formattedFullString"], nOptions=len(candCols),variant=args.model_variant)
  elif args.probing_type=="nli":
    preds = NLIgenerate(devPrepped["formattedFullString"],variant=args.model_variant)
    devPrepped["answer_for_nli"] = preds
    devPrepped["formattedFullString"] = devPrepped.apply(
      _formatExample,cols=cols,probing_type="post_nli",axis=1)
    preds = MCProbeGPT3(devPrepped["formattedFullString"],nOptions=len(candCols),variant=args.model_variant)
  else:
    raise ValueError("Incorrect --probing_type, should be 'lm' or 'mc'")

  preds["pred"] = preds.idxmax(axis=1)
  # preds are in letters (ABC), which map to the order in candCols
  answerMap = dict(zip("ABCD",candCols))

  devPrepped = pd.concat([devPrepped,preds],axis=1)
  
  devPrepped["predAnswer"] = devPrepped[["pred"]+candCols].apply(mapPred,answerMap=answerMap,axis=1)

  accs = computeAccuracies(devPrepped,args)
  if args.output_accuracies_file:
    accs.to_csv(args.output_accuracies_file)
  
  if args.output_prediction_file:
    print("Exporting prediction results to "+args.output_prediction_file)
    #devPrepped.to_pickle(args.output_prediction_file)
    devPrepped.to_csv(args.output_prediction_file)

    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--task",default="siqa",help="tomi or siqa")
  
  parser.add_argument("--instructions",default="")
  # parser.add_argument("--use_cherrypicked",action="store_true")
  parser.add_argument("--model_variant",default="ada")
  
  parser.add_argument("--n_examples",default=3,type=int)
  parser.add_argument("--examples_of_same_type",help="whether to grad in-context examples of the same type (column in data file)")
  parser.add_argument("--stratify_examples_by",help="whether to stratify examples by some type/column in the data file")

  parser.add_argument("--dont_normalize_answer_capitalization",action="store_false",
                      default=True,dest="normalize_answer_capitalization")

  parser.add_argument("--normalize_uncond_prob",action="store_true",
                      help="(lm-probing only) whether to divide by uncong prob before taking argmax. See §2.4 of GPT3 paper.")

  parser.add_argument("--probing_type",help="'lm' for feeding each answer in and picking highest prob or "
                      "'mc' for multiple choice setup where model predict answer letter.")
  
  parser.add_argument("--debug_dev",type=int)
  parser.add_argument("--debug_trn",type=int)

  parser.add_argument("--choose_example_type",help="Column name that will filter the training examples")
  parser.add_argument("--output_prediction_file")
  parser.add_argument("--output_accuracies_file")

  parser.add_argument("--input_prediction_file")

  parser.add_argument("--group_accuracy_by",nargs="+")
  args = parser.parse_args()
  print(args)
  main(args)
