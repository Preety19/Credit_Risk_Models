Header: Model Version
<Model Version>,<Release Date>,<Author>,<Test Result>,<Validation Result>
V2,20-06-2023,Mitesh_Shingare


Header: Description
Jacob frye gives the LGD based on the inpt default rates provided and with the help of additional macro economic factors under consideration.


Header: Performence Kpi
[{'Roc_Auc_Curve': 0.6511, 'r_square': 0.7413}]


Header: Input Parameters
[
{name : ASSESSMENT_DATE,type : object}, 
{name : COLLECTIVE_POOL_ID , type : int64},
{name : DEFAULT_RATE, type : float64},
{name : CYCLE_ID , type : float64},
{name : Assesment Year , type : int64},
{name : Economic Factor Param , type : object},
{name : ECF Year , type : int64},
{name : Multiply by Minus One , type : object},
{name : Parameter Value , type : float64},
{name : Pending Auth , type : object},
{name : ACCOUNT_NO , type : object},
{name : PD , type : object},
{name : LGD , type : object},
{name : COLLECTIVE_POOL_ID , type : object}
]

Header: Output Parameters
[{name:modelOutput, type:float}]


Header: Libraries
[pandas, numpy, sklearn]