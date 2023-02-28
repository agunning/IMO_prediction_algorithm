import math
import random
import numpy
#in which I write some code to predict IMO scores from partial data using something that's approximately MCMC and BayesElo
def FromFile(filename):
    file=open(filename,"r")
    output=[]
    for line in file:
        words=line.split("\t")
        output.append(words)
    file.close()
    return output
numberofyears=4
currentyear=2018
for i in range(0,numberofyears):
    Data[i]=FromFile("imodata"+str(currentyear-i-1)+".txt")

    
for i in range(0,numberofyears):
    Contestants=len(Data)
    ContestantElo=[0]*Contestants # Elo projection of contestants
    PtElo=[0]*42 #Elo projection of question/marks
    Results=[]
    for i in range(0,Contestants):
        NewResult=[]
        for j in range(0,6):
            if Data[i][j+2] not in ["0","1","2","3","4","5","6","7"]:
                NewResult.append(-1)
            else:
                NewResult.append(int(Data[i][j+2]))
                Results.append(NewResult)
    GameContestant=[]
    GamePt=[]
    GameResult=[]
    ContestantGames=[1]*Contestants
    PtGames=[1]*42
    for i in range(0,Contestants):
        for j in range(0,6):
            Result=Results[i][j]
            ContestantGames[i]=ContestantGames[i]+min(7,Result+1)
            if(Result!=-1):
                for k in range(0, Result):
                    GameContestant.append(i)
                    GamePt.append(7*j+k)
                    GameResult.append(1)
                    PtGames[7*j+k]+=1;
                    if Result!=7:
                        GameContestant.append(i)
                        GamePt.append(7*j+Result)
                        GameResult.append(0)
                        PtGames[7*j+Result]+=1;
        GameContestant.append(i)#This gives priors about contestants ability
        GamePt.append(-1)
        GameResult.append(1)
        GameContestant.append(i)
        GamePt.append(-1)
        GameResult.append(0)
    for j in range(0,42):
        GameContestant.append(-1)#This gives priors about question difficulty
        GamePt.append(j)
        GameResult.append(1)
        GameContestant.append(-1)
        GamePt.append(j)
        GameResult.append(0)
    QuestionPriorPrecision=0.5
    ContestantPriorPrecision=0.5
    Games=len(GameResult)
    Temperature=1
    print(PtGames)
    print(Games)
    print(len(GamePt))
    for k in range(0,1000000):
        Game=random.randint(0,Games-1)
        #print(Game)
        Pt=GamePt[Game]
        Contestant=GameContestant[Game]
        if k%10000==9999:
            Temperature=10000/(k+10001)
        if Pt==-1:
            EloDiff=-ContestantElo[Contestant]
            Shock=math.exp(ContestantPriorPrecision*EloDiff)
            Move=ContestantPriorPrecision*Temperature*(GameResult[Game]-1/(Shock+1))
            ContestantElo[Contestant]+=Move
        elif Contestant==-1:
            EloDiff=PtElo[Contestant]
            Shock=math.exp(QuestionPriorPrecision*EloDiff)
            Move=QuestionPriorPrecision*Temperature*(GameResult[Game]-1/(Shock+1))
            PtElo[Pt]-=Move
        else:
            EloDiff=PtElo[Pt]-ContestantElo[Contestant]
            Shock=math.exp(EloDiff)
            Move=Temperature*(GameResult[Game]-1/(Shock+1))
            PtElo[Pt]-=Move
            ContestantElo[Contestant]+=Move
        NewPtElo=[]
        for i in range(0,42):
            NewPtElo.append(round(2000 + 200*PtElo[i]))
            NewContestantElo=[]
        for i in range(0, Contestants):
            NewContestantElo.append(round(2000+200*ContestantElo[i]))
                            
        print(NewPtElo)



        ranking = numpy.argsort(NewContestantElo, )[::-1]


        for i in range(0,Contestants):
            print(NewContestantElo[ranking[i]],Data[ranking[i]][10],Data[ranking[i]][8],Data[ranking[i]][1],Data[ranking[i]][0],sep = " ") #Here's our MLE for contestant skill

    for k in range(0,1000000):
        Game=random.randint(0,Games-1)
        #print(Game)
        Pt=GamePt[Game]
        Contestant=GameContestant[Game]
        if Pt==-1:
            EloDiff=-ContestantElo[Contestant]
            Shock=math.exp(EloDiff)
            Move=ContestantPriorPrecision*Temperature*(GameResult[Game]-1/(Shock+1))
            ContestantElo[Contestant]+=Move
        elif Contestant==-1:
            EloDiff=PtElo[Contestant]
            Shock=math.exp(EloDiff)
            Move=QuestionPriorPrecision*Temperature*(GameResult[Game]-1/(Shock+1))
            PtElo[Pt]-=Move
        else:
            EloDiff=PtElo[Pt]-ContestantElo[Contestant]
            Shock=math.exp(EloDiff)
            Move=Temperature*(GameResult[Game]-1/(Shock+1))
            PtElo[Pt]-=Move
            ContestantElo[Contestant]+=Move
        PtNoise=PtGames[Pt]/Temperature
        PtElo[Pt]+=random.gauss(0,math.sqrt(random.expovariate(PtNoise)))
        ContestantNoise=ContestantGames[Contestant]/Temperature
        ContestantElo[Contestant]+=random.gauss(0,math.sqrt(random.expovariate(ContestantNoise)))

