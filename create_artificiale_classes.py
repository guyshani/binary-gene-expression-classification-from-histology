import pandas as pd
import random
import numpy as np

number_of_genes = 200

a ="TCGA-AG-3896	TCGA-DC-4745	TCGA-CM-6169	TCGA-D5-6931	TCGA-SS-A7HO	TCGA-AZ-6605	TCGA-CA-6717	TCGA-AD-6965	TCGA-AH-6903	TCGA-AH-6643	TCGA-AG-3881	TCGA-CL-5917	TCGA-AG-3902	TCGA-CI-6624	TCGA-AA-A02H	TCGA-AA-A022	TCGA-A6-5662	TCGA-G4-6311	TCGA-CK-4951	TCGA-DC-6683	TCGA-DC-6682	TCGA-F5-6702	TCGA-AF-4110	TCGA-CK-6748	TCGA-CA-5255	TCGA-CK-4952	TCGA-AZ-4315	TCGA-AZ-6603	TCGA-G4-6626	TCGA-DM-A1D7	TCGA-CM-4744	TCGA-F5-6861	TCGA-AG-A02N	TCGA-AF-6136	TCGA-EI-6512	TCGA-AF-6672	TCGA-EI-6514	TCGA-AF-6655	TCGA-AF-A56K	TCGA-G5-6641	TCGA-DY-A1DG	TCGA-AG-A026	TCGA-EI-7004	TCGA-AG-3878	TCGA-CM-5863	TCGA-DM-A280	TCGA-D5-6530	TCGA-A6-6137	TCGA-DM-A1D6	TCGA-A6-2681	TCGA-F4-6463	TCGA-AZ-6599	TCGA-CM-6171	TCGA-AA-A01T	TCGA-CM-5860	TCGA-G4-6628	TCGA-AA-3864	TCGA-F4-6856	TCGA-DM-A0X9	TCGA-AZ-6598	TCGA-A6-A567	TCGA-CA-5796	TCGA-5M-AAT6	TCGA-CM-6163	TCGA-AD-6889	TCGA-CM-4751	TCGA-D5-6924	TCGA-D5-7000	TCGA-D5-6540	TCGA-D5-6932	TCGA-CK-5912	TCGA-AA-3989	TCGA-5M-AAT4	TCGA-A6-6649	TCGA-A6-3809	TCGA-A6-5667	TCGA-AY-4070	TCGA-G4-6299	TCGA-AA-A02O	TCGA-NH-A50T	TCGA-AU-6004	TCGA-DM-A28E	TCGA-A6-5665	TCGA-G4-6315	TCGA-F4-6459	TCGA-D5-6926	TCGA-CK-6747	TCGA-DM-A1D8	TCGA-AA-3858	TCGA-A6-2685	TCGA-NH-A6GA	TCGA-D5-6537	TCGA-G4-6302	TCGA-AG-A01N	TCGA-DT-5265	TCGA-AH-6547	TCGA-AF-A56L	TCGA-DC-4749	TCGA-AA-3846	TCGA-AU-3779	TCGA-CM-5341	TCGA-AA-3844	TCGA-AA-3952	TCGA-D5-5537	TCGA-A6-2686	TCGA-AA-3837	TCGA-D5-6922	TCGA-CM-6674	TCGA-DM-A28A	TCGA-CK-5914	TCGA-G4-6320	TCGA-QL-A97D	TCGA-CM-5868	TCGA-AA-3693	TCGA-D5-5541	TCGA-A6-5657	TCGA-AA-3812	TCGA-CK-4950	TCGA-D5-6541	TCGA-QG-A5YV	TCGA-DM-A1HA	TCGA-AA-A02Y	TCGA-CM-4752	TCGA-CM-6680	TCGA-AD-A5EK	TCGA-AZ-6600	TCGA-CM-6164	TCGA-D5-6898	TCGA-F4-6704	TCGA-CM-5864	TCGA-A6-2671	TCGA-NH-A6GB	TCGA-NH-A8F8	TCGA-CM-6677	TCGA-AA-3949	TCGA-DM-A1D9	TCGA-D5-6930	TCGA-CA-5797	TCGA-D5-6531	TCGA-AZ-4615	TCGA-CM-4743	TCGA-AF-2687	TCGA-AG-3883	TCGA-CL-4957	TCGA-AG-A01Y	TCGA-AG-3890	TCGA-AG-A011	TCGA-AG-3892	TCGA-AF-A56N	TCGA-EI-6513	TCGA-EI-6882	TCGA-F5-6812	TCGA-AG-3726	TCGA-EI-6917	TCGA-AG-A02X	TCGA-AG-A015	TCGA-AF-2693	TCGA-AG-4022	TCGA-AG-A008	TCGA-EI-6509	TCGA-EI-6506	TCGA-CI-6622	TCGA-NH-A50U	TCGA-D5-6928	TCGA-DM-A28K	TCGA-CM-5344	TCGA-G4-6309	TCGA-CM-6165	TCGA-AY-5543	TCGA-CM-6167	TCGA-AA-3664	TCGA-AA-3877	TCGA-D5-6535	TCGA-CM-6678	TCGA-CM-4746	TCGA-A6-6652	TCGA-A6-5661	TCGA-5M-AATE	TCGA-DM-A1D0	TCGA-AA-3681	TCGA-D5-5538	TCGA-CM-6679	TCGA-AA-3818	TCGA-AD-5900	TCGA-A6-6654	TCGA-AA-3814	TCGA-AG-3901	TCGA-AG-3894	TCGA-AG-4015	TCGA-AG-3727	TCGA-AZ-6606	TCGA-CK-4947	TCGA-CA-6716	TCGA-NH-A6GC	TCGA-DC-6681	TCGA-EI-6885	TCGA-DC-5337	TCGA-F5-6464	TCGA-AG-A016	TCGA-F5-6810	TCGA-AG-3898	TCGA-EI-6884	TCGA-AG-A002	TCGA-AG-A00C	TCGA-DY-A0XA	TCGA-F5-6814	TCGA-AA-3848	TCGA-AA-3684	TCGA-F4-6807	TCGA-AA-3875	TCGA-G4-6297	TCGA-A6-5660	TCGA-AZ-4308	TCGA-AG-A01W	TCGA-DC-6155	TCGA-AA-3975	TCGA-AA-3666	TCGA-AA-3979	TCGA-G4-6588	TCGA-CM-5348	TCGA-AA-3833	TCGA-AA-3867	TCGA-AA-A00N	TCGA-CK-5915	TCGA-AA-A03F	TCGA-DM-A1DB	TCGA-CM-6170	TCGA-DM-A285	TCGA-D5-6920	TCGA-F5-6813	TCGA-AG-A020	TCGA-EI-6511	TCGA-AG-A01L	TCGA-AF-2690	TCGA-DC-5869	TCGA-D5-6534	TCGA-AA-3685	TCGA-AA-3851	TCGA-DM-A282	TCGA-F4-6805	TCGA-AA-3821	TCGA-F4-6460	TCGA-AZ-5407	TCGA-EF-5830	TCGA-AA-3984	TCGA-A6-5664	TCGA-AM-5821	TCGA-AA-3956	TCGA-DM-A28F	TCGA-A6-6142	TCGA-G4-6304	TCGA-F4-6806	TCGA-AY-A8YK	TCGA-D5-5539	TCGA-AZ-6601	TCGA-AA-3994	TCGA-G4-6307	TCGA-NH-A5IV	TCGA-QG-A5Z2	TCGA-AA-A01X	TCGA-AY-6197	TCGA-AA-3968	TCGA-AA-3673	TCGA-AA-3856	TCGA-DM-A28M	TCGA-CK-5913	TCGA-CM-6166	TCGA-AA-A01P	TCGA-DM-A28G	TCGA-CA-5256	TCGA-CM-6675	TCGA-AA-3678	TCGA-DC-6158	TCGA-F5-6465	TCGA-DC-6157	TCGA-AG-3909	TCGA-EI-6883	TCGA-EI-6508	TCGA-EI-6881	TCGA-AG-4008	TCGA-AG-3887	TCGA-AG-4021	TCGA-AA-3986	TCGA-CM-5862	TCGA-CK-5916	TCGA-CL-5918	TCGA-AG-3885	TCGA-DY-A1DC	TCGA-EI-6507	TCGA-DC-6154	TCGA-A6-6648	TCGA-AA-A01Z	TCGA-WS-AB45	TCGA-CM-5861	TCGA-A6-6138	TCGA-AM-5820	TCGA-AA-3845	TCGA-AA-A02F	TCGA-3L-AA1B	TCGA-AZ-5403	TCGA-G4-6323	TCGA-AA-3811	TCGA-AA-3831	TCGA-AD-6963	TCGA-CA-6718	TCGA-G4-6306	TCGA-F4-6854	TCGA-DC-6160	TCGA-AH-6544	TCGA-AG-3882	TCGA-AD-6890	TCGA-A6-3807	TCGA-A6-6653	TCGA-AA-A01R	TCGA-G4-6294	TCGA-CM-5349	TCGA-AA-A01V	TCGA-AZ-4682	TCGA-AA-3842	TCGA-AY-A71X	TCGA-F4-6809	TCGA-CM-6172	TCGA-AD-A5EJ	TCGA-AA-3966	TCGA-D5-6927	TCGA-D5-6536	TCGA-D5-6539	TCGA-F4-6461	TCGA-AD-6895	TCGA-CA-6715	TCGA-AA-3850	TCGA-AA-3715	TCGA-AD-6964	TCGA-G4-6586	TCGA-4N-A93T	TCGA-AA-3696	TCGA-AA-3679	TCGA-AA-A03J	TCGA-F4-6703	TCGA-F4-6808	TCGA-CM-6162	TCGA-AD-6888    TCGA-A6-A56B	TCGA-A6-4105	TCGA-AY-6386	TCGA-AY-4071	TCGA-F4-6569	TCGA-AA-3976	TCGA-AD-6899	TCGA-A6-6651	TCGA-D5-6533	TCGA-CK-6746	TCGA-DM-A288	TCGA-A6-4107	TCGA-AA-3947	TCGA-AA-3950	TCGA-AA-3971	TCGA-CK-6751	TCGA-F4-6570"

samples = a.split("\t")
len(samples)

# create a list with pseudo gene names
gene_names = ["gene"+str(i+1) for i in list(range(0,number_of_genes))]
# create a data frame with sample names
col_names = ['symbols']+samples
df = pd.DataFrame(columns= col_names)

# create random classes
for g_name in gene_names:

    # choose if class high or low
    coin = random.random()
    if coin > 0.7:
        # if low is the bigger class
        high_class_num = random.randint(20,179)
        low_class_num = 359 -high_class_num
    else:
        # if high is the bigger class
        low_class_num = random.randint(20,179)
        high_class_num = 359 - low_class_num

    # create sample name list for both classes
    high_class = random.sample(set(samples), high_class_num)
    #low_class = [samples[j] for j in list(range(len(samples))) if samples[j] not in high_class ]

    current_gene = ["high" if name in high_class else "low" for name in samples]
    current_gene = [g_name]+current_gene
    df = df.append(dict(zip(col_names, current_gene)), ignore_index = True)

df.to_csv("/home/guysh/Documents/servers_data/dgx/oncogenes/other/artificiale_label_matrix.csv", index=False)




'''
high = np.array(high)
low = 359 - high
sum(low)/sum(high)
'''
