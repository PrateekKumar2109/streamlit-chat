"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import ReadTheDocsLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain.chains import ConversationChain
from langchain.llms import OpenAI,Cohere
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings,CohereEmbeddings
from langchain.chains import ChatVectorDBChain
import pickle


raw_documents='Rig ABAN-III was deployed and positioned at R_7A platform on 22.07.2019 and well R_7A#1 was spudded on 04.08.2019.    \n30” conductor was pre-piled up to 145m. \nPhase wise drilling is as follows:\n\n	26” Hole Section (0-300m): \nThe well R_7A#1 was spudded on 04.08.2019 at 0730hrs and tagged bottom (sea bed) at 110m. 17 ½” pilot hole (HLB SDMM + MWD) was drilled up to 303m with RAW mud. The hole was enlarged with 26” hole opener up to 303m with Gel mud of 9.2ppg. A wiper trip was carried out and mud weight was increased from 9.2ppg to 9.5ppg. After conditioning the well, 25 joints of 20” casing (X-56, 133ppf, LEOPARD) was lowered with casing shoe at 300m. Cementation was carried out with 185bbl of gel cement slurry of 13.6ppg followed by 210bbl of neat cement slurry of 15.2ppg. W.O.C.   \n	17 ½” Hole Section (300-918m):  \nThe drill string of 17 ½” bit was run in with slick assembly and tagged cement top at 244m. Casing Integrity Test (CIT) was carried out and 20” casing was tested at 500psi in sea water and found ok. The well volume was changed over to SOBM of 9.0ppg and slick assembly was pulled out up to surface. Further 17 ½” PDC bit was run in with HLB directional assembly (RSS+MWD) up to 300m and resumed drilling of 17 ½” hole and drilled down to 919m (SOBM: 9.0-9.7ppg). Pumped hi-vis with increased mud weight from 9.7ppg to 9.8ppg and carried out wiper trip followed by round trip and conditioned the well thoroughly. After conditioning the well thoroughly lowered 13 3/8” casing (J-55, 68ppf, BTC) and kept casing shoe at 918m and float collar at 892m.  Cementation was carried out with 152bbl cement slurry of 15.2ppg and a plug hitting pressure of 1200psi. W.O.C.\n	12 ¼” Hole Section (918-1634m): \nThe drill string with 12 ¼” PDC bit was run in with HLB directional assembly (RSS+LWD) and tagged bottom at 894m (F/C top). Carried out CIT and tested 13 3/8” casing at 1700psi with 9.0ppg of SOBM and found ok. Resumed drilling of 12 ¼” section and drilled down from 892m to 958m with SOBM mud system (MW: 9.0-9.1ppg). At this depth observed poor ROP and pulled out 12 ¼” bit up to surface and further re run 12 ¼” bit with RSS+LWD up to 958m and resumed drilling and drilled down to 1634m (SOBM: 9.1-9.5ppg). After conditioned the well, lowered 9 5/8” casing (L-80, 47ppf, BTC) with 141 joints up to 1634m and kept casing shoe at 1634m and float collar at 1607m. Cementation was carried out with 170bbl cement slurry of 15.8ppg and a plug hitting pressure of 2200psi. W.O.C.\n	8 ½” Hole Section (1634-1777m): \nIn the next run, the drill string with 8 ½” slick assembly was run in and tagged bottom at 1607m. Casing of 9 5/8” was tested @ 3500psi and found ok. Resumed 8 ½” hole drilling and drilled float collar and cement up to 1632m. The well volume was changed over from sea water to 9.2ppg of NDDF mud system. Further resumed drilling of 8 ½” hole and drilled down cement, casing shoe and fresh formation up to 1636m. At this depth it was decided to cut an 18m conventional core and core was cut from 1636m to 1640m, while coring observed poor ROP and further coring was terminated at 1640m. Core was pulled up to surface and recovered CC#1 from 1636m to 1640m (Recovery: 2.04m, 51%). Further run in 8 ½” PDC bit with RSS+LWD up to 1636m and enlarged cored portion from 1636m to 1640m and further drilling of 8 ½” hole was continued and drilled down to 1672m. At this depth observed drill string stalling due to high torque. Further drilling stopped at this depth and it was decided to cut an 18m conventional core. Further run in core bit with core barrel up to 1672m and cut a conventional core from 1672m to 1676.12m. After cutting of core up to 1676.12m further coring was terminated due to poor ROP. Broke off and pulled out core up to surface and observed empty core barrel and CC#2 recovery is nil (only few broken fragments of Limestone). Further run in with 8 ½” PDC bit with slick assembly up to 1677.12m by clearing held up portion and drilled down up to 1677.12m. Further drilling was stopped and at this depth it was decided to cut a conventional core. Further run in core bit with core barrel up to 1677.12m and cut a conventional core from 1677.12m to 1684.34m and further coring was stopped due to poor ROP. Broke off and pulled out core barrel up to surface and recovered CC#3 from 1677.12-1684.34m (Recovery: 5.13m; 71%). \nFurther run in with 8 ½” PDC bit with HLB DIR assembly (RSS+LWD) and enlarged cored portion and drilled up to 1685m. Observed HLB tool gave erratic value, rectified the same but no success. Resumed drilling and drilled down from 1685 to 1765m (While drilling observed dynamic loss @ 5-15bbl/hr.). At this depth observed high torque and string got stalled continuously. Pumped hi-vis and circulate out. Resumed drilling and drilled 8 ½” from 1739 to 1777m (Rev. TD of the well) and while drilling observed dynamic mud loss @ 9bbl/hr. and static loss nil. Pumped tandem hi-vis pill and circulate out with increased mud weight from 9.2ppg to 9.3ppg with controlled discharge to avoid mud loss prior to wiper trip. Carried out wiper trip followed by round trip and conditioned the well thoroughly. After that LithoScanner + CMR-Gr log and DSI-FMI-Gr log were recorded by Schlumberger. DSI log was recorded both in 8 ½” open hole section as well as behind 9 5/8” casing. After conditioning the well 7” liner (L-80, 29ppf) casing was lowered with keeping casing shoe at 1777m, L/Collar at 1739m & Hanger Top at 1489m.  \nCementation was carried out with 65bbl cement slurry of 15.8ppg. After W.O.C. run in with 6” bit up to 1489m (H/Top) and tested Hanger top at 3500psi and was dressed.  Further drilled cement and cleared the well up to 1739m (L/Collar top). Carried out tandem scrapper trip and scrapped the well up to 1739m. CBL-VDLGR-CCL log was recorded from 1736-1489m @ 700psi in NDDF mud system.  \nAfter recording of log well volume was changed over to sea water and 7” casing was tested hermetically at 3500psi on 11.10.2019 at 0600hrs.   \n   \n1.1.	WELL COMPLETION & ACTIVATION\n\nThe 7” liner was tested hermetically @ 3500psi and found Ok. Change over the well volume with 8.6ppg filtered brine. Mukta pay was perforated in the interval 1675-1668m (7m) @ 5spf with 4 ½” STIM gun in two runs. After perforation, run in 3 ½” tubing (EUE, L-80, 9.3ppf) with 7” hydraulic packer up to 1613m and set packer at 1613m. Packer tested at 2500psi and found holding. Tested the same through annulus @ 1000 psi and found Ok. Carried out injectivity test and found injectivity 0.55 BPM at 2000spi. Vessel Ocean Turquoise reported at site on 27.10.2019 at 1130hrs and carried out NEMAJ job by Ocean Turquoise vessel and pumped 20bbl of sea water + 30bbl preflush + 50bbl of ERA+N2 solution + 30bbl over flush and recorded maximum pumping pressure 2352psi @ 2.50 BPM. Well-kept for soaking ~2.5hrs and recorded STHP= 1850psi. After NEMAJ job open the well opened through 1/8” choke & gradually increased to ½” choke size and observed well flowing spent acid with thick oil and gas. Recorded FTHP @ 390psi & FTHT=34°C.  Well kept-closed from 0630hrs to 1000hrs on 28.10.2019 for Mechanical work (STHP= 500psi) and further opened the well on 28.10.2019 at 1000hrs and gradually increased choke size from ½” to ¾” and observed flow of thick oil and gas with FTHP= 240psi and FTHT= 45°C. Continued well flowed through ¾” choke and observed flow of foamy oil and gas with FTHP= 265psi and FTHT= 47°C. Further well flowed through separator with 1” choke and observed flow of oil @ 2500 BOPD and gas @ 8325 m3/day (due to foamy nature of oil and increase of gas in oil line, oil rate cannot be considered accurate). Closed the well and flushed out separator. Further opened the well through 1” choke bypass separator and observed flow of foamy oil and gas with FTHP= 230psi and FTHT- 56°C.  Continued well flowed oil @ 2657 BOPD and gas @ 9430 m3/day (Oil & Gas rate not accurate, turbine reading with foamy oil) with 1 ¼” choke through separator and observed FTHP= 190psi and FTHT= 59°C. \nFurther closed the well on 31.10.2019 at 1230hrs and tested reservoir lubricator at 1000psi and found ok. Run in hole 1.5” dummy up to 1610m and pull out up to surface. Run in EMG tool up to 1605m and recorded SBHP and temperature and pull out EMG tool up to surface after one hour. Again run in EMG tool up to 1605m for recording of FBHP. Opened the well with 16/64” choke to 1 ¼” choke size and observed flow of foamy oil and gas with FTHP= 460-225psi and FTHT= 34-56°C. Recorded FBHP and temperature and pull out EMG tool up to surface. Closed the well and retrieved EMG data. \nAgain lowered EMG tool up to 1605m for FBHP study. Opened the well through burner with 16/64” choke and gradually increased to ¼” choke size for flow stabilization. Observed well flowing foamy oil @ 2550 BOPD and gas @ 9571 m3/day (foamy oil) with FTHP= 190psi, FTHP= 54°C and H2S= 5-20ppm. Further well shut in for buildup study on 01.11.2019 at 1130hrs and observed STHP= 510psi. Continued well kept-closed since 02.11.2019 at 1130hrs for recording of SBHP and temperature gradient at 1605m. Pull out EMG tool up to surface and retrieved EMG data. \nRun in oil sample barrel up to 1400m and pull out up to surface and observed oil sample not collected properly. Again run in with oil sample barrel up to 1400m and collected two sample. After pull out sample barrel up to surface, observed samples collected successfully.  \nThe well was bulldozed 1.5 times of string volume with brine of 8.6ppg. Well-kept under observation for tubing puncture job prior to lowering of ESP completion and monitored static losses and observed static loss @ 30bbl/hr. Run in 1 11/16” dummy up to 1612.3m and pull out up to surface. Run in Schlumberger perforation gun of 1 11/16” (7shot) Wireline up to 1612.43m and punctured tubing at 1609m successfully. After pulled out perforation gun carried out circulation and observed static loss @ 30bph and dynamic loss @ 36bph. After N/dn X-mass tree and N/up BOP. Tested BOP and found ok. After N/up and testing of BOP, unseat packer and pull out 3 ½” tubing along with 7” packer up to surface. Run in completion string (3 ½” EUE tubing) and ESP with controlled speed along with chemical injection line up to 1040m. Installed cable clamp in every joints and run in dummy up to 1015m. Further run in 3 ½” EUE tubing (L-80, 9.3ppf) and ESP with controlled speed with power cable and chemical injection line up to 1063m. Installed cable clamp in every joints. Continued Novomet work on packer and carried out insulation test below the packer and connected packer with power cable. Carried out connectivity test and found ok. Resumed and run in 3 ½” EUE tubing and ESP with controlled speed up to 1602m and ESP cable connected with tubing hanger and tested ok. \nTested tubing hanger and tubing hanger seal at 2500psi and found ok. N/dn BOP and N/up X-mass tree. Tested X-mass tree valve @ 4500psi in stages and found ok. Rig up slick line and run in 2.75” dummy followed by standing valve up to 561m and set 9 5/8” hydraulic packer at 548m and tested in stages @ 3500psi and found ok. \nRig down slick line and connected ESP cable with platform panel and tested ESP reading and found ok. After ESP optimization open the well through stab burner with 1” choke and initially observed flow of brine only later well flowed oil and water with FTHP= 380psi, Salinity= 31344as NaCl. Continued flow back through 3/8” choke with ESP and observed well flowing oil @ 1174 BOPD (observed 1000ml foamy oil = 800ml), gas @ 1013 m3/day and recorded H2S = 21ppm. ESP parameters: Pi= 1965psi, PD: 2249psi and Tm= 96°C. Well R-7A#1 diverted and handed over to platform on 13.11.2019 at 1700hrs. Rig ABAN-III skidded from well R-7A#1 to R-7A#7 on 13.11.2019 at 1800hrs. \nraw_documents',

# Split text
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_text(raw_documents)


# Load Data to vectorstore
embeddings = CohereEmbeddings(cohere_api_key= "vGCEakgncpouo9Nz0rsJ0Bq7XRvwNgTCZMKSohlg")
#embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(documents, embeddings)

def load_chain(vectorstore):
    """Logic for loading the chain you want to use should go here."""
    llm = Cohere(model="command-xlarge-nightly", cohere_api_key="vGCEakgncpouo9Nz0rsJ0Bq7XRvwNgTCZMKSohlg",temperature=0)
    chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore)
    #chain = ConversationChain(llm=llm)
    return chain




chain = load_chain(vectorstore)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Chatbot", page_icon=":shark:")
st.header("ChatBot Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Write me a linkedin post?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
