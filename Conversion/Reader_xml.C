// root -l -b Reader_xml.C'("Radion","low","2016")'
#include "TMVA/Reader.h"

void Reader_xml(TString signal, TString mass, TString y){

  TString flashgg_data = "/eos/user/l/lata/Resonant_bbgg/flattrees_legacybranch_detaHHvar_6thFeb2020/";
  TString indirMVA;
  TString dirOut;
  TString weightFile;

  for (TString year : {y}){
    Float_t MVAscaling;


    indirMVA = flashgg_data + year + "/"; 
    dirOut = "analysistrees/";
    weightFile = "/eos/user/l/lata/Resonant_bbgg/output_training_fromEOStrees_" + signal + "_" + mass+"mass_" + year + "/" + mass+"mass_XGB_training_file.weights.xml"; 
    MVAscaling=1.0;
    

    //For calculation MVA transformation
    TString MVAFlatteningFileName = "/eos/user/l/lata/Resonant_bbgg/output_training_fromEOStrees_" + signal + "_" + mass+"mass_" + year + "/cumulativeTransformation_Total_preselection_diffNaming.root";
    TFile * MVAFlatteningFile = new TFile(MVAFlatteningFileName,"READ");
    TGraph * MVAFlatteningCumulative = (TGraph*)MVAFlatteningFile->Get("cumulativeGraph"); 
    TString Sig1, Sig2, Sig3, Sig4, Sig5, Sig6;
    
    if(mass == "low"){
      Sig1 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-260_narrow_13TeV-madgraph.root";
      Sig2 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-270_narrow_13TeV-madgraph.root";
      Sig3 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-280_narrow_13TeV-madgraph.root";
      Sig4 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-300_narrow_13TeV-madgraph.root";
      Sig5 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-320_narrow_13TeV-madgraph.root";
      Sig6 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-350_narrow_13TeV-madgraph.root";
    }
    else if(mass == "mid"){
      Sig1 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-400_narrow_13TeV-madgraph.root";
      Sig2 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-450_narrow_13TeV-madgraph.root";
      Sig3 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-500_narrow_13TeV-madgraph.root";
      Sig4 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-550_narrow_13TeV-madgraph.root";
      Sig5 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-600_narrow_13TeV-madgraph.root";
      Sig6 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-650_narrow_13TeV-madgraph.root";
    }
    else{
      Sig1 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-700_narrow_13TeV-madgraph.root";
      Sig2 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-800_narrow_13TeV-madgraph.root";
      Sig3 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-900_narrow_13TeV-madgraph.root";
      Sig4 = "output_GluGluTo" + signal + "ToHHTo2B2G_M-1000_narrow_13TeV-madgraph.root";
 }

    TString Sig = "output_GluGluTo" + signal + "ToHHTo2B2G_M-" +mass + "mass.root";
  
 
    TString Bkg1 ="output_DiPhotonJetsBox_MGG-80toInf_13TeV-Sherpa.root";
    TString Bkg2, Bkg3, Bkg7;
    if(year=="2016") Bkg2 ="output_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCUETP8M1_13TeV_Pythia8.root";
    if(year=="2016") Bkg3 ="output_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCUETP8M1_13TeV_Pythia8.root";
    if(year=="2017" || year=="2018") Bkg2 ="output_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8.root";
    if(year=="2017" || year=="2018") Bkg3 ="output_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8.root";
    TString Bkg4 ="output_GluGluHToGG_M-125_13TeV_powheg_pythia8.root";
    TString Bkg5 ="output_VBFHToGG_M-125_13TeV_powheg_pythia8.root";
    TString Bkg6 ="output_VHToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8.root";
    if(year=="2016" or year=="2017") Bkg7 ="output_ttHToGG_M125_13TeV_powheg_pythia8.root";
    else Bkg7 = "output_ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8.root";
    TString Bkg8 ="output_bbHToGG_M-125_4FS_yb2_13TeV_amcatnlo.root";
    TString Bkg9 ="output_bbHToGG_M-125_4FS_ybyt_13TeV_amcatnlo.root";
    TString Data ="Data.root";

//=================================================================================================================

    //for (TString fname : {Sig,Sig1,Sig2,Sig3,Sig4,Bkg1,Bkg2,Bkg3,Bkg4,Bkg5,Bkg6,Bkg7,Bkg8,Bkg9,Data}){
      for (TString fname : {Sig,Sig1,Sig2,Sig3,Sig4,Bkg1,Bkg2,Bkg3,Bkg4,Bkg5,Bkg6,Bkg7,Bkg8,Bkg9,Data,Sig5,Sig6}){
  // create TMVA::Reader object
	TMVA::Reader *reader = new TMVA::Reader("!V:!Silent:Color,G:AnalysisType=multiclass");

  // create a set of variables and declare them to the reader - the variable names must corresponds in name and type to
  // those given in the weight file(s) that you use
	Float_t var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15, var16, var17, var18, var19, var20, var21, var22;

	reader->AddVariable( "absCosThetaStar_CS", &var1 );
	reader->AddVariable( "absCosTheta_bb", &var2 );
	reader->AddVariable( "absCosTheta_gg", &var3 );
	reader->AddVariable( "PhoJetMinDr", &var4 );
	reader->AddVariable( "PhoJetOtherDr", &var5 );
	reader->AddVariable( "customLeadingPhotonIDMVA", &var6 );
	reader->AddVariable( "customSubLeadingPhotonIDMVA", &var7 );
	reader->AddVariable( "leadingJet_DeepFlavour", &var8 );
	reader->AddVariable( "subleadingJet_DeepFlavour", &var9 );
	reader->AddVariable( "leadingPhotonSigOverE", &var10 );
	reader->AddVariable( "subleadingPhotonSigOverE", &var11 );
	reader->AddVariable( "sigmaMOverM", &var12 );
	reader->AddVariable( "diphotonCandidatePtOverdiHiggsM", &var13 );
	reader->AddVariable( "dijetCandidatePtOverdiHiggsM", &var14 );
	reader->AddVariable( "(leadingJet_bRegNNResolution*1.4826)", &var15);
	reader->AddVariable( "(subleadingJet_bRegNNResolution*1.4826)", &var16 );
	reader->AddVariable( "(sigmaMJets*1.4826)", &var17 );
	reader->AddVariable( "leadingPhoton_pt/CMS_hgg_mass", &var18 );
	reader->AddVariable( "subleadingPhoton_pt/CMS_hgg_mass", &var19);
	reader->AddVariable( "leadingJet_pt/Mjj", &var20 );
	reader->AddVariable( "subleadingJet_pt/Mjj", &var21 );
	reader->AddVariable( "rho", &var22 );

  // book the MVA of your choice (prior training of these methods, ie, existence of the weight files is required)
  //reader->BookMVA( "BDT", "training_with_extended_var_29_11_2019_year2016.weights.xml");
	reader->BookMVA( "BDT", weightFile);

	TString fnameAll = indirMVA + fname;
	TFile *inputAll = TFile::Open(fnameAll);
	
	Float_t absCosThetaStar_CS, absCosTheta_bb, absCosTheta_gg, PhoJetMinDr, customLeadingPhotonIDMVA, customSubLeadingPhotonIDMVA, leadingJet_DeepFlavour, subleadingJet_DeepFlavour, leadingPhotonSigOverE, subleadingPhotonSigOverE, sigmaMOverM, diphotonCandidatePtOverdiHiggsM, dijetCandidatePtOverdiHiggsM, leadingJet_bRegNNResolution, subleadingJet_bRegNNResolution, sigmaMJets, CMS_hgg_mass, leadingPhoton_pt, subleadingPhoton_pt, Mjj, leadingJet_pt, subleadingJet_pt, PhoJetOtherDr, rho, deltaEtaHH;
	TTree* resTree;
	if (fname == "Data.root") resTree = (TTree*)inputAll->Get("tagsDumper/trees/Data_13TeV_DoubleHTag_0");
	else{ 
	  resTree = (TTree*)inputAll->Get("tagsDumper/trees/bbggtrees_13TeV_DoubleHTag_0");
	}
	//resTree->Print();

	resTree->SetBranchAddress("absCosThetaStar_CS",&absCosThetaStar_CS);
	resTree->SetBranchAddress("absCosTheta_bb",&absCosTheta_bb);
	resTree->SetBranchAddress("absCosTheta_gg",&absCosTheta_gg);
	resTree->SetBranchAddress("PhoJetMinDr",&PhoJetMinDr);
	resTree->SetBranchAddress("customLeadingPhotonIDMVA",&customLeadingPhotonIDMVA);
	resTree->SetBranchAddress("customSubLeadingPhotonIDMVA",&customSubLeadingPhotonIDMVA);
	resTree->SetBranchAddress("leadingJet_DeepFlavour",&leadingJet_DeepFlavour);
	resTree->SetBranchAddress("subleadingJet_DeepFlavour",&subleadingJet_DeepFlavour);
	resTree->SetBranchAddress("leadingPhotonSigOverE",&leadingPhotonSigOverE);
	resTree->SetBranchAddress("subleadingPhotonSigOverE",&subleadingPhotonSigOverE);
	resTree->SetBranchAddress("sigmaMOverM",&sigmaMOverM);
	resTree->SetBranchAddress("diphotonCandidatePtOverdiHiggsM",&diphotonCandidatePtOverdiHiggsM);
	resTree->SetBranchAddress("dijetCandidatePtOverdiHiggsM",&dijetCandidatePtOverdiHiggsM);
	resTree->SetBranchAddress("leadingJet_bRegNNResolution",&leadingJet_bRegNNResolution);
	resTree->SetBranchAddress("subleadingJet_bRegNNResolution",&subleadingJet_bRegNNResolution);
	resTree->SetBranchAddress("sigmaMJets",&sigmaMJets);
	resTree->SetBranchAddress("CMS_hgg_mass",&CMS_hgg_mass);
	resTree->SetBranchAddress("leadingPhoton_pt",&leadingPhoton_pt);
	resTree->SetBranchAddress("subleadingPhoton_pt",&subleadingPhoton_pt);
	resTree->SetBranchAddress("Mjj",&Mjj);
	resTree->SetBranchAddress("leadingJet_pt",&leadingJet_pt);
	resTree->SetBranchAddress("subleadingJet_pt",&subleadingJet_pt);
	resTree->SetBranchAddress("PhoJetOtherDr",&PhoJetOtherDr);
	resTree->SetBranchAddress("rho",&rho);
	resTree->SetBranchStatus("*", 1);

	TString fileout= dirOut + fname;
	TFile *target = new TFile(fileout,"RECREATE" );
	TTree *outTree=resTree->CloneTree();
	Float_t BDT_response;
	Float_t mvaScaled;
	double flatMVA;
	TBranch *xmlMVA = outTree->Branch("xmlMVA",&BDT_response,"xmlMVA/F");
	TBranch *xmlMVAtransf = outTree->Branch("xmlMVAtransf",&flatMVA,"xmlMVAtransf/D");

	cout<<"Start MVA calculation...."<<endl;

	TStopwatch sw;
	sw.Start();

	for (Long64_t ievt=0; ievt<resTree->GetEntries();ievt++) {
	  resTree->GetEntry(ievt);
	  
	  if (ievt%10000 == 0){
	    cout << "--- ... Processing event: " << ievt << endl;
	  }
	  
	  var1 = absCosThetaStar_CS;
	  var2 = absCosTheta_bb;
	  var3 = absCosTheta_gg;
	  var4 = PhoJetMinDr;
	  var6 = customLeadingPhotonIDMVA;
	  var7 = customSubLeadingPhotonIDMVA;
	  var8 = leadingJet_DeepFlavour;
	  var9 = subleadingJet_DeepFlavour;
	  var10 = leadingPhotonSigOverE;
	  var11 = subleadingPhotonSigOverE;
	  var12 = sigmaMOverM;
	  var13 = diphotonCandidatePtOverdiHiggsM;
	  var14 = dijetCandidatePtOverdiHiggsM;
	  var15 = leadingJet_bRegNNResolution*1.4826;
	  var16 = subleadingJet_bRegNNResolution*1.4826;
	  var17 = sigmaMJets*1.4826;
	  var18 = leadingPhoton_pt/CMS_hgg_mass;
	  var19 = subleadingPhoton_pt/CMS_hgg_mass;
	  var20 = leadingJet_pt/Mjj;
	  var21 = subleadingJet_pt/Mjj;
	  var5 = PhoJetOtherDr;
	  var22 = rho;
	  //var23 = deltaEtaHH;
	  
	  //retrieve the corresponding MVA output
	  BDT_response=(reader->EvaluateMulticlass("BDT"))[2];  //Signal
	  mvaScaled = BDT_response/(BDT_response*(1.-MVAscaling)+MVAscaling);
	  flatMVA = MVAFlatteningCumulative->Eval(mvaScaled);     
 
	  xmlMVA->Fill(); 
	  xmlMVAtransf->Fill(); 
	}
	// get elapsed time
	sw.Stop();
	sw.Print();
	outTree->Write();
	MVAFlatteningFile->Close();
	target->Close(); 
	delete reader;
	inputAll->Close();
	
	cout<<"Finish MVA calculation."<<endl;
      }

 } 

}
 
