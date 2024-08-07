import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
#from countrycodes import countrycodes
#in which I write some code to predict IMO scores from partial data using something that's approximately MCMC and BayesElo
headings = ["Name", "Country", "Q1","Q2","Q3","Q4","Q5","Q6","Total", "Rank", "Award"]
def binary_partial_down(x):
    output=[]
    i=0
    while x!=0:
        if ((2**i)&x)!=0:
            output.append(x)
            x-=2**i
        i+=1
    return output
def binary_partial_up(x):
    output=[]
    i=0
    while x!=256:
        if ((2**i)&x)!=0:
            output.append(x)
            x+=2**i
        i+=1
    return output
    
def index_q(x):
    if x[1].isnumeric():
        return int(x[1])-1
    else:
        return x
def name_q(x):
    if isinstance(x , int):
        return "Q"+str(x+1)
    else:
        return x
def ordinal(x):
    if x%100 in range(10,20):
        return str(x)+"th"
    if x%10 == 1:
        return str(x)+"st"
    if x%10 == 2:
        return str(x)+"nd"
    if x%10 == 3:
        return str(x)+"rd"
    return str(x)+"th"
def toCsv(filename,newfilename=""):
    file=open(filename,"r")
    output=[]
    for line in file:
        words=line.split("\t")
        output.append(words)
    file.close()
   
   
    newDF = pd.DataFrame(output, columns = headings)
    if (newfilename!=""):
      newDF.to_csv(newfilename, index=False)
    return newDF
def scramble(filename,newfilename="",filetype='xml',Scteam=0,Scind=0,Trials=1000):
    '''
e    Function used for testing, creates a typical partial set of data available from a known complete IMO in the past
    SCteam: the number of (country, problem) pairs not yet marked
    SCInd: the number of (individual, problem) pairs not yet marked (in practice this number will be very small, for problematic scripts, and this probably does leak some information about what the values in question are)
    Additionally some scores are obscured despite their values being known to prevent people from doing precisely the thing I'm doing now.
    Any of the XMLs given should work here and give you a csv of what this should typically look like.
    '''
    '''
    TODO: fix for CSVs
    '''
    if filetype=="xml":
        rawdata=pd.read_xml(filename).sort_values('code').reset_index()
        rawdata['Name']=rawdata.name+" "+rawdata.surname
        rawdata['Country']=rawdata.code
        data=np.array([rawdata['problem'+str(i)] for i in range(1,7)]).T

    else:
        rawdata=pd.read_csv(filename)
        rawdata.rename(columns = lambda x: index_q(x), inplace=True)
        data = rawdata[range(0,6)].astype("Int64", copy=False)
    deletedRows=[0]*6
    for i in range(0,len(data)):
        if i==0 or rawdata.Country[i]!=rawdata.Country[i-1]: #That is, if the contestant is from a nation not previously seen
            k=0
            omit=np.random.permutation(6) #Usually, the markers remove a different mark for each candidate
            for j in range(0,6):
                if random.random()<Scteam:
                    deletedRows[j]=1
                else:
                    deletedRows[j]=0
        for j in range(0,6):
            if deletedRows[j]==1:
                data[j][i]=-1
            elif random.random()<Scind:
                data[j][i]=-1
            elif omit[k]==j:
                data[j][i]=-1
        k+=1
    print(rawdata)
    data.rename(columns = lambda x: name_q(x), inplace=True)
    for i in range(6):
        rawdata[name_q(i)]=data[name_q(i)]
    
    print(rawdata.columns)
    if newfilename!="":
        #daa.to_csv(newfilename, index=False)
         rawdata.drop(["Total","Rank","Award",0,1,2,3,4,5],axis=1).to_csv(newfilename, index=False)
    return data


def predict(filename,
            interpret= False,
            tfilename="",
            candeval=False,
            filetype="csv",
            countrydp=0,
            countryvar=0,
            problemdp=0,
            problemvar=0,
            problemmu=0,
            Trials=1000,
    
):
    '''
    tfilename: if this is a path to known complete data known this gives calibration data on predictions, if it is the empty string it does nothing
    interpret: gives summary statistics for a complete IM
    filetype: 'xml' or 'csv'
    candeval: takes complete data and gives individual contestant Elos; to keep things fair the contributions of countries are set to 0

    returns an IMOPredictionOutput object where you can run
    printCuts(): explains medal cutoffs
    printElos(): if candeval is set to true prints candidate elos
    plotProblem(): takes a number from 1 to 6 and gives you a plot of how hard the problem is
    plotContest(): the same but for the whole contest
    printWinner(): explains a probability distribution over the likely winners
    countryResult(): takes a 3 letter country code
    plotCalibration() gives you a calibration plot of predictions this algorithm makes vs a known baseline,
    The predictions in question are everything of the form "with probability P candidate X gets score Y on problem Z"
    The output of any of these functions is generally self-explanatory.
    
    '''
    '''
    1000 trials should suffice to get cutoffs, 10000 for results for individual countries
    '''
    if filetype=="xml":
        data=pd.read_xml(filename).sort_values('code').reset_index()
        data['Name']=data.name+" "+data.surname
        data['Country']=data.code
        Results=np.array([data['problem'+str(i)] for i in range(1,7)]).T

    elif filetype=="csv":
        data=pd.read_csv(filename)
        data.rename(columns = lambda x: index_q(x), inplace=True)
        Results = data[range(0,6)].astype("Int64", copy=False).T
    else:
        raise Exception("filetype must be csv or xml")
    if tfilename!="": 
        calibration=True
        tdata=pd.read_csv(tfilename)
        tdata.rename(columns = lambda x: index_q(x), inplace=True)
        tResults  = tdata[range(0,6)].astype("Int64", copy=False).T
    else:
        calibration=False
    Contestants = len(data)
    InfoParticipants=[]
    InfoType=[]
    ContestantInfo=[1]*Contestants
    PtInfo=[1]*42
    for i in range(0,Contestants):
        for j in range(0,6):
            Result=Results[i][j]
            #print(Result.dtype)
            if(not Result==-1):
                ContestantInfo[i]=ContestantInfo[i]+min(7,Result+1)
                for k in range(0, Result):
                    InfoParticipants.append([i,7*j+k])
                    InfoType.append(1)
                    PtInfo[7*j+k]+=1;
                if Result!=7:
                    InfoParticipants.append([i,7*j+Result])
                    InfoType.append(0)
                    PtInfo[7*j+Result]+=1;
        InfoParticipants.append(i)#This gives priors about contestants ability
        InfoType.append(2)
    for j in range(0,42):
        InfoParticipants.append(j)
        InfoType.append(3)
    QuestionPriorPrecision=1
    ContestantPriorPrecision=1
    QuestionMean=0
    ContestantElo=[0]*Contestants
    ContestantPrior=[0]*Contestants
    ContestantEloDeltasq=[0]*Contestants
    PtElo=[0]*42
    Temperature=1
 
    EContestantElo=[0]*Contestants
    EContestantEloSq=[0]*Contestants
    EPtElo=[0]*42
    EPtEloSq=[0]*42
    contestantsGreaterThan=np.zeros((42,Contestants+1))
    if calibration == True :
        RealResults=[]
        PredictedResults=[]
        for i in range(0, Contestants):
            for j in range(0,6):
                if np.isnan(Results[i][j]):
                    RealResults.append(tResults[i][j])
                    PredictedResults.append([0]*8)
 
    if interpret==False:
        GoldCuts=[0]*43
        SilverCuts=[0]*43
        BronzeCuts=[0]*43
        MaxScores=[0]*43
        PerfectScores=[0]*100
    if candeval==False:
        Countries=pd.unique(data.Country)
        countriesGreaterThan=np.zeros((252,len(Countries)+1))
      
        CRankings={}
        CScores={}
        for Country in Countries:
            CRankings[Country]=[0]*len(Countries)
            CScores[Country]=np.zeros(253)

            CMedals={}
        for Country in Countries:
            CMedals[Country]={}#there are like 210 different ways a country could get medals, and most of them will never happen, so let's store this as a dict.
        CIndex={}
        for i in range(0,len(Countries)):
            CIndex[Countries[i]]=i
        CContList=[[] for Country in Countries]
        for i in range(0,Contestants):
            CContList[CIndex[data.Country[i]]].append(i)
        CElo=[0]*len(Countries)
        ECElo=[0]*len(Countries)
        ECEloSq=[0]*len(Countries)
        
        for i in range(0,len(Countries)):
            InfoParticipants.append(i)
            InfoType.append(5)
        CPriorPrecision=1
    if True:
        InfoParticipants.append(-1)
        InfoType.append(4)
    Infos=len(InfoType)
    TrialLength=10000
    trialno=0
    print("Running trials:")
    for k in tqdm(range(0,TrialLength*Trials+1000000)):
        Info=random.randint(0,Infos-1)
        #print(Game)
        Type = InfoType[Info]
        Pt=-1
        Contestant=-1
        #print(Type)
        if Type==2:
            Contestant=InfoParticipants[Info]
            EloDiff=ContestantElo[Contestant]-ContestantPrior[Contestant]
            Move=(ContestantPriorPrecision+CPriorPrecision)*Temperature*EloDiff
            ContestantElo[Contestant]-=Move
        elif Type==3:
            Pt=InfoParticipants[Info]
            EloDiff=PtElo[Pt]-QuestionMean
            Move=QuestionPriorPrecision*Temperature*EloDiff
            PtElo[Pt]-=Move
        elif (Type==4 and k>500000):
            QuestionPriorPrecision=np.random.gamma((43+problemdp)/2,2/(((np.array(PtElo)-np.array(QuestionMean))**2).sum()+1+problemdp*(problemvar+(problemmu-QuestionMean)**2))+.000001)
            ContestantPriorPrecision=np.random.gamma((Contestants+1)/2,2/(((np.array(ContestantElo)-np.array(ContestantPrior))**2).sum()+1)+.01)
            QuestionMean=random.gauss((42*np.array(PtElo).mean()+problemdp*problemmu)/(42+problemdp),1/math.sqrt(QuestionPriorPrecision*(42+problemdp)))
            
            if (candeval == False and k>1000000):
                #print(CElo)
                CPriorPrecision=np.random.gamma((len(Countries)+1+countrydp)/2,2/((np.array(CElo)**2).sum()+1+countryvar*countrydp)+.000001)
            #print( QuestionPriorPrecision,ContestantPriorPrecision,CPriorPrecision,QuestionMean)
        elif (Type==5 and k>100000):
            Country=InfoParticipants[Info]
            CContElos = [ContestantElo[Contestant] for Contestant in CContList[Country]]
            CElo[Country]=random.gauss(
                sum(CContElos)*ContestantPriorPrecision/(len(CContElos)*ContestantPriorPrecision+CPriorPrecision),
                math.sqrt(1/(len(CContElos)*ContestantPriorPrecision+CPriorPrecision)))
            #print(CElo[Country])
            
            for Contestant in CContList[Country]:
                ContestantPrior[Contestant]=CElo[Country]
        elif (Type!=5) and (Type!=4):
            Contestant=InfoParticipants[Info][0]
            Pt=InfoParticipants[Info][1]
            EloDiff=PtElo[Pt]-ContestantElo[Contestant]
            #print(EloDiff)
            Shock=np.exp(EloDiff)
            Move=Temperature*(InfoType[Info]-1/(Shock+1))
            PtElo[Pt]-=Move
            ContestantElo[Contestant]+=Move
        if Pt!=-1:
            PtNoise=PtInfo[Pt]/Temperature
            PtElo[Pt]+=random.gauss(0,math.sqrt(1/PtNoise))
        if Contestant!=-1:
            ContestantNoise=ContestantInfo[Contestant]/Temperature
            ContestantElo[Contestant]+=random.gauss(0,math.sqrt(1/ContestantNoise))
        if k%10000==9999 and k<=1000000:
            Temperature= 10000/(k+1)
            #print(Temperature)
        elif k%TrialLength==TrialLength-1 and k>500000: #We generate a possible IM0
            
            #print("hi")
            for i in range(0,Contestants):
                EContestantElo[i]+=(2000+ContestantElo[i]*200)/Trials
                EContestantEloSq[i]+=(2000+ContestantElo[i]*200)**2/Trials
                ContestantEloDeltasq[i] += (ContestantElo[i]-ContestantPrior[i])**2/Trials
            for i in range(0,42):
                EPtElo[i]+=(2000+PtElo[i]*200)/Trials
                EPtEloSq[i]+=(2000+PtElo[i]*200)**2/Trials
            if (candeval==False):
                for i in range(0,len(Countries)):
                    ECElo[i]+=(2000+CElo[i]*200)/Trials
                    ECEloSq[i]+=(2000+CElo[i]*200)**2/Trials
            if (interpret==False):
                ContestantScores=[0]*Contestants
                ContestantHM=[0]*Contestants    
                m=0             
                MaxScore=-1
                PerfectScore=0
                for i in range(0, Contestants):
                    ContElo=ContestantElo[i]
                    Score=0
                    
                    for j in range(0,6):
                        if not(Results[i][j]==-1):
                            Score+=Results[i][j]
                            if (Results[i][j]==7):
                                ContestantHM[i]=1
                        else: #We simulate a contestant working on a problem
                            jj=0
                            while jj<7:
                                if random.random()>1/(1+np.exp(PtElo[7*j+jj]-ContElo)):
                                    break
                                jj+=1
                            Score+=jj
                            if jj==7:
                                ContestantHM[i]=1
                            if calibration==True:
                                PredictedResults[m][jj]+=100/Trials
                                m+=1
                            
                    ContestantScores[i]=Score
                    MaxScore=max(Score,MaxScore)
                    if Score==42:
                        PerfectScore+=1
            #print(NewResults)
            
            if(interpret==False):
                def update_country_medal_count(Bronze,Silver,Gold,p):
                    CMedalCount={}
                    for Country in Countries:
                        CMedalCount[Country]=[0]*4
                    for i in range(0,Contestants):
                        Country=data.Country[i]
                        Score=ContestantScores[i]
                        if Score >= Gold:
                            CMedalCount[Country][0]+=1
                        elif Score >= Silver:
                            CMedalCount[Country][1]+=1
                        elif Score >= Bronze:
                            CMedalCount[Country][2]+=1
                        elif ContestantHM[i]==1:
                            CMedalCount[Country][3]+=1
                    for Country in Countries:
                        Medals=CMedalCount[Country]
                        MedalCode = (8**3)*Medals[0]+(8**2)*Medals[1]+(8**1)*Medals[2]+Medals[3]
                        if MedalCode not in CMedals[Country].keys():
                            CMedals[Country][MedalCode]=100*p/Trials
                        else:
                            CMedals[Country][MedalCode]+=100*p/Trials
                    return 0
                ScoreDistribution=[0]*43
                for i in range(0,Contestants):
                    ScoreDistribution[ContestantScores[i]]+=1
                contestantsGT=Contestants-np.cumsum(ScoreDistribution)[:42]
                for i in range(0,42):
                    contestantsGreaterThan[i,contestantsGT[i]]+=1
                bcu = np.searchsorted(-contestantsGT,-Contestants/2) #bcl=bcu-1
                pub=(contestantsGT[bcu-1]-Contestants/2)/(contestantsGT[bcu-1]-contestantsGT[bcu])
                BronzeCuts[bcu+1]+=pub*100/Trials
                BronzeCuts[bcu]+=(1-pub)*100/Trials
                #if upper bronze cuts
                bc=bcu
                pb=pub
                scu = np.searchsorted(-contestantsGT,-contestantsGT[bc]/2) #bcl=bcu-1
                pus=(contestantsGT[scu-1]-contestantsGT[bc]/2)/(contestantsGT[scu-1]-contestantsGT[scu])
                SilverCuts[scu+1]+=pb*pus*100/Trials
                SilverCuts[scu]+=pb*(1-pus)*100/Trials
                gcu = np.searchsorted(-contestantsGT,-contestantsGT[bc]/6) #bcl=bcu-1
                pug=(contestantsGT[gcu-1]-contestantsGT[bc]/6)/(contestantsGT[gcu-1]-contestantsGT[gcu])
                GoldCuts[gcu+1]+=pb*pug*100/Trials
                GoldCuts[gcu]+=pb*(1-pug)*100/Trials
                update_country_medal_count(bc+1,scu+1,gcu+1,pb*pus*pug)
                update_country_medal_count(bc+1,scu,gcu+1,pb*(1-pus)*pug)
                update_country_medal_count(bc+1,scu+1,gcu,pb*pus*(1-pug))
                update_country_medal_count(bc+1,scu,gcu,pb*(1-pus)*(1-pug))
                #if lover bronze cuts
                bc=bcu-1
                pb=1-pub
                scu = np.searchsorted(-contestantsGT,-contestantsGT[bc]/2) #bcl=bcu-1
                pus=(contestantsGT[scu-1]-contestantsGT[bc]/2)/(contestantsGT[scu-1]-contestantsGT[scu])
                SilverCuts[scu+1]+=pb*pus*100/Trials
                SilverCuts[scu]+=pb*(1-pus)*100/Trials
                gcu = np.searchsorted(-contestantsGT,-contestantsGT[bc]/6) #bcl=bcu-1
                pug=(contestantsGT[gcu-1]-contestantsGT[bc]/6)/(contestantsGT[gcu-1]-contestantsGT[gcu])
                GoldCuts[gcu+1]+=pb*pug*100/Trials
                GoldCuts[gcu]+=pb*(1-pug)*100/Trials
                update_country_medal_count(bc+1,scu+1,gcu+1,pb*pus*pug)
                update_country_medal_count(bc+1,scu,gcu+1,pb*(1-pus)*pug)
                update_country_medal_count(bc+1,scu+1,gcu,pb*pus*(1-pug))
                update_country_medal_count(bc+1,scu,gcu,pb*(1-pus)*(1-pug))
                MaxScores[MaxScore]+=100/Trials
                PerfectScores[PerfectScore]+=100/Trials
                CTotalScore={}
                for Country in Countries:
                    CTotalScore[Country]=0
                for i in range(0,Contestants):
                    Country=data.Country[i]
                    CTotalScore[Country]+=ContestantScores[i]
                QuickScoreDist=[0]*256
                CRanking={}
                for Country in Countries:
                    for t in binary_partial_down(CTotalScore[Country]):
                        QuickScoreDist[t]+=1
                    CRanking[Country]=0
                for Country in Countries:
                    for t in binary_partial_up(CTotalScore[Country]+1):
                        CRanking[Country]+=QuickScoreDist[t]
                    CRankings[Country][CRanking[Country]]+=100/Trials
                    CScores[Country][CTotalScore[Country]]+=100/Trials
                a=np.bincount(list(CTotalScore.values()),minlength=253)
                b=len(Countries)-np.cumsum(a)[:252]
                for i in range(252):
                    countriesGreaterThan[i,b[i]]+=1
               
    output = IMOPredictionOutput()
    if interpret==False:
        output.cuts=cutsStatistics(GoldCuts,SilverCuts,BronzeCuts,MaxScores,PerfectScores)
        output.countryRankings=CRankings   
        output.countryMedals=CMedals
        output.countryScores=CScores
        output.countriesGreaterThan=countriesGreaterThan/Trials
    if calibration==True:
        output.predictions=PredictedResults
        output.results=RealResults
    if True:
        output.numContestants=Contestants
        output.contestantsGreaterThan=contestantsGreaterThan/Trials
        output.contestantEloDeltasq=ContestantEloDeltasq
        contestantdata = data[["Name", "Country"]].copy()
        contestantdata["Elo"]=np.array(EContestantElo)
        contestantdata["EloSD"]=(np.array(EContestantEloSq)-np.array(EContestantElo)**2)**0.5
        output.contestantElos=contestantdata
        output.problemElos=[np.array(EPtElo),(np.array(EPtEloSq)-np.array(EPtElo)**2)**0.5]
        if candeval==False:
            output.countryElos=pd.DataFrame([[Countries[i],ECElo[i],(np.array(ECEloSq[i])-np.array(ECElo[i])**2)**0.5] for i in range(0,len(Countries))],columns=['Country','Elo','EloSD'])
    print(QuestionPriorPrecision)
    print(ContestantPriorPrecision)
    print(QuestionMean)
    print(CPriorPrecision)
    print(trialno)
    return output

class EloOutput:
    def __init__(self,names,elo,eloSD):
        self.names=names
        self.elo=elo
        self.eloSD=eloSD
class cutsStatistics:
    def __init__(self,GoldCuts,SilverCuts,BronzeCuts,MaxScores,PerfectScores):
        self.GoldCuts=GoldCuts
        self.SilverCuts=SilverCuts
        self.BronzeCuts=BronzeCuts
        self.MaxScores=MaxScores
        self.PerfectScores=PerfectScores
    
    def printCuts(self,deluxe=False):                    
        print("Here are my predictions for this year's medal cutoffs")
        print("Gold cuts:")
        for i in range(0,43):
            if self.GoldCuts[i]>0.5:
                print(i,"  ",round(self.GoldCuts[i]), "% of all simulations", sep="")
        print("Silver cuts:")
        for i in range(0,43):
            if self.SilverCuts[i]>0.5:
                print(i,"  ", round(self.SilverCuts[i]), "% of all simulations", sep = "")
        print("Bronze cuts:")
        for i in range(0,43):
            if self.BronzeCuts[i]>0.5:
                print(i,"  ",round(self.BronzeCuts[i]),"% of all simulations", sep = "")
        if (round(self.PerfectScores[0],1) == 100):
            print("There were no simulations in which a perfect score was achieved")
        else:
            if(self.PerfectScores[0]==0):
                print("A perfect score was achieved in every simulation")
            else:
                print("No perfect score was achieved in "+str(round(self.PerfectScores[0]))+"% of simulations")
                print("Here's the probabilities for the number of 42s when there was at least one:")
            for i in range(0,len(self.PerfectScores)):
                if self.PerfectScores[i]>0.5:
                    print(str(i)+"  "+str(round(self.PerfectScores[i]))+"%")
        if (self.PerfectScores[0]!=0.5):
            print("The highest mark achieved was:")
            for i in range(0,43):
                if self.MaxScores[i]>0.5:
                    print(str(i)+"  "+str(round(self.MaxScores[i]))+"%")                

class IMOPredictionOutput:
    
    def __init__(self):
        self.cuts=None
        self.countryRankings=None
        self.countryMedals=None
        self.predictions=None
        self.results=None
        self.contestantElos=None
        self.countryElos=None
        self.problemElos=None
        self.contestantEloDeltasq=None
    def printCuts(self):
        return self.cuts.printCuts()

    def printElos(self):
        print("Here is a list of candidates sorted by their (computed) Elos. Note that these numbers are ridiculously noisy.")
        elodata = self.contestantElo
        print( elodata.sort_values(by=["Elo"], ascending = False).to_string())
        return 0
    def plotProblem(self,problem):
        X=list(range(500,3500,50))
        problemdata=output["problemElos"]
        prob=np.array([[0.]*8]*len(X))
        for i in range(0,len(X)):
            q=1
            culm=0
            for j in range(0,7):
                p=1/(1+math.exp((X[i]-problemdata[0][7*(problem-1)+j])/200))
                prob[i][j]=q*p
                q*=1-p
                prob[i][7]=q
                df = pd.DataFrame(prob, columns=range(0,8))
                df["Elo"] = X
                df
                plt.close()
                df.plot.area(colormap="YlOrBr",x="Elo",title = "Score probabilities by Elo")
                plt.show()
        return df
    def plotContest(self):
        X=list(range(500,3500,50))
        problemdata=self.problemElos
        pprob=np.array([[[0.]*43]*6]*len(X))

        for i in range(0,len(X)):
            for k in range(0,6):
                q=1
                culm=0
                for j in range(0,7):
                    p=1/(1+math.exp((X[i]-problemdata[0][7*k+j])/200))
                    pprob[i][k][j]=q*p
                    q*=1-p
                    pprob[i][k][7]=q
                    fftpprob=np.array([
                        np.array([np.fft.fft(pprob[i][k]) for k in range(0,6)]).T
            for i in range(0,len(X))])
                    prob =np.abs( [np.fft.ifft([np.prod(fftpprob[i][j]) for j in range(0,43)]) for i in range(0,len(X))])
                    df = pd.DataFrame(prob, columns=range(0,43))
                    df["Elo"] = X
        def do_label(x):
            if (not isinstance(x,int)) or (x%7==0) or (x%7==3):
                return x
            else:
                return "_"+str(x)
            df2=df.copy()
            df2.rename(columns = lambda x: do_label(x), inplace=True)
            plt.close()
            #sns.set()
        graph = df2.plot.area(colormap="nipy_spectral",x="Elo",title = "Score probabilities by Elo")
        plt.show()
        return df
    def CountryResult(self,country):
        #if country in countrycodes.keys():
        #    country=countrycodes[country]
        if country not in self.countryRankings.keys():
            print(country+" not in database for this IMO")
            return -1
        rankings=self.countryRankings[country]
        medals=self.countryMedals[country]
        culm=np.cumsum(rankings)

        print( "In 80% of simulations " + country+ "'s team came between "+ordinal(np.searchsorted(culm,10)+1)+" and "+ordinal(np.searchsorted(culm,90)+1)+".")
        print( "The median performance was "+ordinal(np.searchsorted(culm,50)+1)+".")
        rmin = np.searchsorted(culm,0.01)
        rmax = np.searchsorted(culm,99.99)
        plt.close()
        plot = plt.bar(list(range(rmin,rmax+1)),rankings[rmin-1:rmax])
        plt.xlabel('Ranking')
        plt.ylabel("Percent Probability")
        plt.title("Distribution of " + country + "'s possible rankings")
        plt.show()


        results = self.countryScores[country]
        cresults = np.cumsum(results)
        cmin = np.searchsorted(cresults,0.01)
        cmax = np.searchsorted(cresults,99.99)
        plt.close()
        plot = plt.bar(list(range(cmin,cmax+1)),results[cmin-1:cmax])
        plt.xlabel('Ranking')
        plt.ylabel("Percent Probability")
        plt.title("Distribution of " + country + "'s possible results")
        plt.show()


        
        print( "The mean score is " + str(np.round(results @ np.arange(253)/100,2))+", the median is " + str(np.searchsorted(cresults, 50)) + '.')
        print( "In 80% of outcomes Australia scored beteern "+str(np.searchsorted(cresults, 10))+ " and "+str(np.searchsorted(cresults, 90))+".")

       

        print( "Here are the most likely medal outcomes for " + country + ":")
 
        
        outcomes= []
        for result in medals.keys():
            outcomes.append([medals[result],result])
            outcomes.sort(key = lambda x: x[0])
            outcomes.reverse()

        cprob=0
        print ("\t| G\t S\t B\t HM")
        for i in range(0,min(len(outcomes),10)):
            print(str(round(outcomes[i][0]))+"% \t| "+
                  str(outcomes[i][1]//(8**3))+"\t "+
                  str((outcomes[i][1]//(8**2))%8)+"\t "+
                  str((outcomes[i][1]//8)%8)+"\t "+
                  str(outcomes[i][1]%8))
            cprob+=outcomes[i][0]
            if cprob>99.5:
                break
        else:
            if len(outcomes)>10:
                print( str(round(100-cprob))+"% \t Other")
    def printWinner(self):
        rankings=self.countryRankings
        print("Here are the countries most likely to win the IMO.\nNote that I count ties as a win for both sides, so the numbers might add up to more than 100%.")
        winners=[]
        for country in rankings.keys():
            if rankings[country][0]>0.5:
                winners.append([rankings[country][0],country])
                winners.sort(key = lambda x: x[0])
                winners.reverse()
        for [prob, country] in winners:
            print(str(round(prob))+"% \t" + country)
    def plotCalibration(self):
        intervals=[0,1,5,10,20,30,50,70,80,90,95,99,100]
        successes=[0]*(len(intervals)-1)
        trials=[0]*(len(intervals)-1)
        predictions=self.predictions
        results=self.results
        for i in range(len(predictions)):
            culm = 0
            k=0
            for j in range(0,7):
                culm+=predictions[i][j]
                while (k!= len(trials)-1) and culm>intervals[k+1]:
                    k+=1
                    print(i,j)
                    trials[k]+=1
                if results[i]<=j:
                    successes[k]+=1
                    print(successes,trials)
                    x=[0.005*(intervals[i]+intervals[i+1]) for i in range(0,len(trials))]
                    y=[(successes[i]+0.0001*x[i])/(trials[i]+0.01) for i in range(0,len(trials))]
                    yerr = [2*((x[i])*(1-(x[i]))/(trials[i]+0.01))**(0.5) for i in range(0, len(trials))]
                    plt.close()
                    plt.errorbar(x, y, yerr, marker='s')
                    plt.plot([0,0],[1,1],'k-',color = 'b')
                    plt.ylim(0,1)
                    plt.xlabel("Expected probability")
                    plt.ylabel("Actual fraction of occurences")
                    plt.show()
        return plt
    def queryContestantScore(self,score):
        culm = np.cumsum(self.contestantsGreaterThan[score])
        med = np.searchsorted(culm,.5)
        print('An individual score of '+str(score)+' was in the median simulation good for a position of '+ordinal(med+1)+ ' (80% confidence interval '+ ordinal(np.searchsorted(culm,.1)+1)+'-'+ordinal(np.searchsorted(culm,.9)+1) +').')
        print("That's as least as good as " + str(round(100-100*med/self.numContestants,1))+"% of contestants")
    def queryCountryScore(self,score):
        culm = np.cumsum(self.countriesGreaterThan[score])
        print('An country score of '+str(score)+' was in the median simulation good for a position of '+ordinal(np.searchsorted(culm,.5)+1)+ ' (80% confidence interval '+ ordinal(np.searchsorted(culm,.1)+1)+'-'+ordinal(np.searchsorted(culm,.9)+1) +').')
        
    
        
