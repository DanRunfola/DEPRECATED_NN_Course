#Load the metadata
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import sys

meta = pd.read_csv("/home/dan/data/Charades_v1_480/Charades_v1_train.csv")

print(meta.head)

def processVideos(meta, metaID, label):
    #First, lookup the videos that have the action of interest
    meta = meta[meta['actions'].str.contains(metaID) == True]
    
    #For each video, create a snippet of the sequence for just that action
    #and output it into the appropriate folder.
    for i, row in meta.iterrows():
        actions = row["actions"].split(";")
        filterActions = [s for s in actions if metaID in s]
        for j in filterActions:
            start = int(round(float(j.split(" ")[1]),0))
            end = int(round(float(j.split(" ")[2]),0))
            print(start)
            print(end)
            ffmpeg_extract_subclip("/home/dan/data/Charades_v1_480/"+row["id"]+".mp4", start, end, targetname="/home/dan/data/refactorCharades/"+label+"/"+row["id"]+".mp4")
        
        
#Drinking
processVideos(meta, metaID="c106", label="Drinking")
processVideos(meta, metaID="c149", label="LaughSmile")
processVideos(meta, metaID="c151", label="Sitting")
processVideos(meta, metaID="c126", label="Throwing")
processVideos(meta, metaID="c097", label="Walking")
