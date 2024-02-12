Header: Model Version
<Model Version>,<Release Date>,<Author>,<Test Result>,<Validation Result>
V2,20-06-2023,Mitesh_Shingare


Header: Description
Vasicek PD Model gives us the Point In Time forward looking PD for the future years to be predicted.


Header: Performence Kpi
[{'Roc_Auc_Curve': 0.7182,'R-square':0.7732}]


Header: Input Parameters
[
{name : ASSESSMENT_DATE,type:object} , 
{name : COLLECTIVE_POOL_ID, type : int64},
{name : SEGMENT_DESC , type : string},
{name : DEFAULT_RATE , type : float64},
{name : Assesment Year , type : int64},
{name : Economic Factor Param , type : string},
{name : ECF Year , type : int64},
{name : Multiply by Minus One , type : string},
{name : Parameter Value , type : float64},
{name : Pending Auth , type : string}
]

Header: Output Parameters
[{name:modelOutput, type:float}]


Header: Libraries
[pandas, numpy, sklearn]