import sys, types, os
from optparse import OptionParser, make_option
from  pprint import pprint
from array import array
#from ROOT import gROOT, TCanvas, TFile, TTree
#from rootpy.io import root_open
#from rootpy.tree import Tree, TreeChain
#from rootpy.plotting import Hist
#from rootpy.extern.six.moves import range

# -----------------------------------------------------------------------------------------------------------
def main(options,args):

    ## setTDRStyle()
    ROOT.gStyle.SetOptStat(0)
        
    fin = ROOT.TFile.Open(options.file)
    tree = fin.Get("reducedTree_sig")

    for nameTagPos,s in enumerate(options.file.split("/")):
        print nameTagPos, s
        if "outfil" in s:
            nameTagPos += 1 
            break

    print nameTagPos
    name = options.file.split("/")[nameTagPos]


    fout = ROOT.TFile.Open("cumulativeTransformation_"+name,"recreate")

    nbins = 10000
    xlow = 0.
    xup = 1.
    histoMVA = ROOT.TH1F("histoMVA","histoMVA",nbins,xlow,xup)
#    tree.Draw("MVAOutput>>histoMVA",ROOT.TCut("weight"))
    tree.Draw("MVAOutput>>histoMVA")
#    histoMVA.FillRandom("gaus",1000000)

    cumulativeHisto = histoMVA.GetCumulative()
    cumulativeHisto.Scale(1./histoMVA.Integral())
    cumulativeGraph = ROOT.TGraph(cumulativeHisto)
    cumulativeGraph.SetTitle("cumulativeGraph")
    cumulativeGraph.SetName("cumulativeGraph")

    evalCumulatives = ROOT.TH1F("eval","eval",nbins/10,0,1)

    x , y = array( 'd' ), array( 'd' )
    step = (xup-xlow)/nbins
    for i in range(1,10000):
#        xvalue = ROOT.gRandom.Gaus()
        xvalue = ROOT.TH1.GetRandom(histoMVA)
        evalCumulatives.Fill(cumulativeGraph.Eval(xvalue))
    evalCumulatives.Sumw2()
    evalCumulatives.Scale(1./evalCumulatives.Integral())
    evalCumulatives.GetYaxis().SetRangeUser(0,2./evalCumulatives.GetNbinsX())

    c = ROOT.TCanvas()
    histoMVA.SetLineColor(ROOT.kRed)
    histoMVA.Draw()


    print name

    formats = [".png",".pdf"]

    for format in formats:
        c.SaveAs(name+"_func"+format)

    cumulativeGraph.Draw("AP")
    for format in formats:
        c.SaveAs(name+"_cum"+format)

    evalCumulatives.Draw("EP")
    for format in formats:
        c.SaveAs(name+"_evalx"+format)
    

    cumulativeGraph.Write()
    fout.Write()
    fout.Close()

    fin.cd()

    processes = [
        "reducedTree_sig",
        "reducedTree_bkg",
        "reducedTree_bkg_0",
        "reducedTree_bkg_2",
        "reducedTree_bkg_3",
        "reducedTree_bkg_4",
        "reducedTree_bkg_5",
        "reducedTree_bkg_6",
        "reducedTree_bkg_7",
        #"reducedTree_bkg_8",
        #"reducedTree_bkg_9",
        #"reducedTree_bkg_10"
        ]

    fin = ROOT.TFile.Open(options.file)
    print fin

    fTransformed = ROOT.TFile.Open(options.file.replace(".root","")+"_transformedMVA.root","recreate")
    print fTransformed


    for proc in processes:
        print proc
        tree = fin.Get(proc)
        #print tree
        #print tree.GetName()
        #chain = ROOT.TChain(tree.GetName())
        if proc=="reducedTree_sig":
          chain = ROOT.TChain("reducedTree_sig")
        elif proc=="reducedTree_bkg":
          chain = ROOT.TChain("reducedTree_bkg")
        elif proc=="reducedTree_bkg_0":
          chain = ROOT.TChain("reducedTree_bkg_0")
        elif proc=="reducedTree_bkg_1":
          chain = ROOT.TChain("reducedTree_bkg_1")
        elif proc=="reducedTree_bkg_3":
          chain = ROOT.TChain("reducedTree_bkg_3")
        elif proc=="reducedTree_bkg_4":
          chain = ROOT.TChain("reducedTree_bkg_4")
        elif proc=="reducedTree_bkg_5":
          chain = ROOT.TChain("reducedTree_bkg_5")
        elif proc=="reducedTree_bkg_6":
          chain = ROOT.TChain("reducedTree_bkg_6")
        elif proc=="reducedTree_bkg_7":
          chain = ROOT.TChain("reducedTree_bkg_7")
        elif proc=="reducedTree_bkg_8":
          chain = ROOT.TChain("reducedTree_bkg_8")
        elif proc=="reducedTree_bkg_9":
          chain = ROOT.TChain("reducedTree_bkg_9")
        elif proc=="reducedTree_bkg_10":
          chain = ROOT.TChain("reducedTree_bkg_10")
        else:
          chain = ROOT.TChain("reducedTree_bkg_2")

    
        chain.Add(options.file)
        copyTree = chain.CopyTree("")
        copyTree.SetName(proc)
        copyTree.SetTitle(proc)

        transfMVA = array( 'f', [ 0. ] )
        transfBranch = copyTree.Branch("MVAOutputTransformed",transfMVA,"MVAOutputTransformed/F");
        dummyList = []
        
        entry = ROOT.TTree.GetEntries
        
        for i,event in enumerate(copyTree):
            if i>ROOT.TTree.GetEntries:break
#            if i>tree.GetEntries():break
            mva = event.MVAOutput
            transfMVA[0] = cumulativeGraph.Eval(mva)
            transfBranch.Fill()
    
    
    fTransformed.Write()
    fTransformed.Close()

        
if __name__ == "__main__":

    parser = OptionParser(option_list=[
            make_option("-i", "--infile",
                        action="store", type="string", dest="file",
                        default="",
                        help="input file",
                        ),
            ]
                          )

    (options, args) = parser.parse_args()
    sys.argv.append("-b")

    
    pprint(options.__dict__)

    import ROOT
    
    main(options,args)
        
