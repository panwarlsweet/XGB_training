#include "TMVA/Reader.h"

void Reader_xml(){

TString flashgg_data = "/eos/user/l/lata/Resonant_bbgg/flattrees_legacybranch_detaHHvar_6thFeb2020/2016/";
TString indirMVA;
TString dirOut;
TString weightFile;

for (TString year : {"2016"}){
Float_t MVAscaling;

 if(year=="2016"){ 
   indirMVA = "/eos/user/l/lata/Resonant_bbgg/flattrees_legacybranch_detaHHvar_6thFeb2020/2016/"; 
   dirOut = "/eos/user/l/lata/Resonant_bbgg/flattrees_legacybranch_detaHHvar_6thFeb2020/test_2016/";
   weightFile = flashgg_data + "lowmass_radion_XGB_training_file_year16.weights.xml"; 
   MVAscaling=1.0;
 }

//For calculation MVA transformation
TString MVAFlatteningFileName = flashgg_data + "cumulativeTransformation_lowmass_radion_nodata_year16.root";  //with Mjj - ETH
TFile * MVAFlatteningFile = new TFile(MVAFlatteningFileName,"READ");
TGraph * MVAFlatteningCumulative = (TGraph*)MVAFlatteningFile->Get("cumulativeGraph"); 

TString Sig = "output_GluGluToRadionToHHTo2B2G_M-250_narrow_13TeV-madgraph.root";
TString Bkg1 ="output_DiPhotonJetsBox_MGG-80toInf_13TeV-Sherpa.root";
TString Bkg2, Bkg3;
if(year=="2016") Bkg2 ="output_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCUETP8M1_13TeV_Pythia8.root";
if(year=="2016") Bkg3 ="output_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCUETP8M1_13TeV_Pythia8.root";
if(year=="2017" || year=="2018") Bkg2 ="output_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8.root";
if(year=="2017" || year=="2018") Bkg3 ="output_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8.root";
TString Bkg4 ="output_GluGluHToGG_M-125_13TeV_powheg_pythia8.root";
TString Bkg5 ="output_VBFHToGG_M-125_13TeV_powheg_pythia8.root";
TString Bkg6 ="output_VHToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8.root";
TString Bkg7 ="output_ttHToGG_M125_13TeV_powheg_pythia8.root";
TString Bkg8 ="output_bbHToGG_M-125_4FS_yb2_13TeV_amcatnlo.root";
TString Bkg9 ="output_bbHToGG_M-125_4FS_ybyt_13TeV_amcatnlo.root";
TString Data ="Data.root";

//=================================================================================================================

for (TString fname : {Sig}){//,Bkg1,Bkg2,Bkg3,Bkg4,Bkg5,Bkg6,Bkg7,Bkg8,Bkg9}){
 
  // create TMVA::Reader object
  TMVA::Reader *reader = new TMVA::Reader("!V:!Silent:Color,G:AnalysisType=multiclass");

  // create a set of variables and declare them to the reader - the variable names must corresponds in name and type to
  // those given in the weight file(s) that you use
  Float_t var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15, var16, var17, var18, var19, var20, var21, var22, var23;

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
  reader->AddVariable( "leadingJet_bRegNNResolution", &var15);
  reader->AddVariable( "subleadingJet_bRegNNResolution", &var16 );
  reader->AddVariable( "sigmaMJets", &var17 );
  reader->AddVariable( "leadingPhoton_pt/CMS_hgg_mass", &var18 );
  reader->AddVariable( "subleadingPhoton_pt/CMS_hgg_mass", &var19);
  reader->AddVariable( "leadingJet_pt/Mjj", &var20 );
  reader->AddVariable( "subleadingJet_pt/Mjj", &var21 );
  reader->AddVariable( "rho", &var22 );
  reader->AddVariable( "deltaEtaHH", &var23 );

  // book the MVA of your choice (prior training of these methods, ie, existence of the weight files is required)
  //reader->BookMVA( "BDT", "training_with_extended_var_29_11_2019_year2016.weights.xml");
  reader->BookMVA( "BDT", weightFile);

  TString fnameAll = indirMVA + fname;
  TFile *inputAll = TFile::Open(fnameAll);

  Float_t absCosThetaStar_CS, absCosTheta_bb, absCosTheta_gg, PhoJetMinDr, customLeadingPhotonIDMVA, customSubLeadingPhotonIDMVA, leadingJet_DeepFlavour, subleadingJet_DeepFlavour, leadingPhotonSigOverE, subleadingPhotonSigOverE, sigmaMOverM, diphotonCandidatePtOverdiHiggsM, dijetCandidatePtOverdiHiggsM, leadingJet_bRegNNResolution, subleadingJet_bRegNNResolution, sigmaMJets, CMS_hgg_mass, leadingPhoton_pt, subleadingPhoton_pt, Mjj, leadingJet_pt, subleadingJet_pt, PhoJetOtherDr, rho, deltaEtaHH;

  TTree *resTree = (TTree*)inputAll->Get("tagsDumper/trees/bbggtrees_13TeV_DoubleHTag_0");
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
      var15 = leadingJet_bRegNNResolution;
      var16 = subleadingJet_bRegNNResolution;
      var17 = sigmaMJets;
      var18 = leadingPhoton_pt/CMS_hgg_mass;
      var19 = subleadingPhoton_pt/CMS_hgg_mass;
      var20 = leadingJet_pt/Mjj;
      var21 = subleadingJet_pt/Mjj;
      var5 = PhoJetOtherDr;
      var22 = rho;
      var23 = deltaEtaHH;
      /*
      var1 = Mjj;
      var2 = leadingJet_DeepFlavour;
      var3 = subleadingJet_DeepFlavour;  
      var4 = absCosThetaStar_CS;
      var5 = absCosTheta_bb;
      var6 = absCosTheta_gg;
      var7 = diphotonCandidatePtOverdiHiggsM;
      var8 = dijetCandidatePtOverdiHiggsM;
      var9 = customLeadingPhotonIDMVA;
      var10 = customSubLeadingPhotonIDMVA;
      var11 = leadingPhotonSigOverE;
      var12 = subleadingPhotonSigOverE;
      var13 = sigmaMOverM;
      var14 = leadingPhoton_pt/CMS_hgg_mass;
      var15 = subleadingPhoton_pt/CMS_hgg_mass;
      var16 = leadingJet_pt/Mjj;
      var17 = subleadingJet_pt/Mjj;
      var18 = rho;
      var19 = leadingJet_bRegNNResolution*1.4826;
      var20 = subleadingJet_bRegNNResolution*1.4826;
      var21 = sigmaMJets*1.4826;
      var22 = PhoJetMinDr;
      var23 = PhoJetOtherDr;
       */
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

 
