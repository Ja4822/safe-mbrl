БИ
╖'Р'
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	АР
A
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
ю
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
PyFunc
input2Tin
output2Tout"
tokenstring"
Tin
list(type)("
Tout
list(type)(И
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
Л
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.15.42v1.15.3-68-gdf8c55cов
n
PlaceholderPlaceholder*
dtype0*
shape:         <*'
_output_shapes
:         <
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:         *
shape:         
h
Placeholder_2Placeholder*
dtype0*
shape:         *#
_output_shapes
:         
h
Placeholder_3Placeholder*
dtype0*#
_output_shapes
:         *
shape:         
h
Placeholder_4Placeholder*
dtype0*
shape:         *#
_output_shapes
:         
h
Placeholder_5Placeholder*
dtype0*#
_output_shapes
:         *
shape:         
h
Placeholder_6Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
N
Placeholder_7Placeholder*
dtype0*
shape: *
_output_shapes
: 
N
Placeholder_8Placeholder*
shape: *
dtype0*
_output_shapes
: 
е
0pi/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:*
valueB"<      
Ч
.pi/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *╛*"
_class
loc:@pi/dense/kernel
Ч
.pi/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*"
_class
loc:@pi/dense/kernel*
valueB
 *>*
_output_shapes
: 
я
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*"
_class
loc:@pi/dense/kernel*
dtype0*

seed*
_output_shapes
:	<А*
seed2*
T0
┌
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@pi/dense/kernel*
_output_shapes
: *
T0
э
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	<А*"
_class
loc:@pi/dense/kernel*
T0
▀
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
:	<А*"
_class
loc:@pi/dense/kernel
й
pi/dense/kernel
VariableV2*"
_class
loc:@pi/dense/kernel*
shape:	<А*
_output_shapes
:	<А*
shared_name *
dtype0*
	container 
╘
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*
T0*
_output_shapes
:	<А*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(

pi/dense/kernel/readIdentitypi/dense/kernel*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	<А
Р
pi/dense/bias/Initializer/zerosConst*
_output_shapes	
:А* 
_class
loc:@pi/dense/bias*
dtype0*
valueBА*    
Э
pi/dense/bias
VariableV2*
shared_name * 
_class
loc:@pi/dense/bias*
_output_shapes	
:А*
shape:А*
	container *
dtype0
┐
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros*
use_locking(*
T0*
_output_shapes	
:А* 
_class
loc:@pi/dense/bias*
validate_shape(
u
pi/dense/bias/readIdentitypi/dense/bias*
_output_shapes	
:А* 
_class
loc:@pi/dense/bias*
T0
Х
pi/dense/MatMulMatMulPlaceholderpi/dense/kernel/read*
T0*
transpose_b( *
transpose_a( *(
_output_shapes
:         А
К
pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*(
_output_shapes
:         А*
data_formatNHWC*
T0
Z
pi/dense/TanhTanhpi/dense/BiasAdd*
T0*(
_output_shapes
:         А
й
2pi/dense_1/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@pi/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ы
0pi/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*$
_class
loc:@pi/dense_1/kernel*
valueB
 *╫│▌╜*
_output_shapes
: 
Ы
0pi/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *╫│▌=*$
_class
loc:@pi/dense_1/kernel*
dtype0*
_output_shapes
: 
Ў
:pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_1/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
АА*

seed*
T0*
seed2*$
_class
loc:@pi/dense_1/kernel
т
0pi/dense_1/kernel/Initializer/random_uniform/subSub0pi/dense_1/kernel/Initializer/random_uniform/max0pi/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_1/kernel*
_output_shapes
: *
T0
Ў
0pi/dense_1/kernel/Initializer/random_uniform/mulMul:pi/dense_1/kernel/Initializer/random_uniform/RandomUniform0pi/dense_1/kernel/Initializer/random_uniform/sub*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
АА*
T0
ш
,pi/dense_1/kernel/Initializer/random_uniformAdd0pi/dense_1/kernel/Initializer/random_uniform/mul0pi/dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
АА*
T0*$
_class
loc:@pi/dense_1/kernel
п
pi/dense_1/kernel
VariableV2*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *$
_class
loc:@pi/dense_1/kernel
▌
pi/dense_1/kernel/AssignAssignpi/dense_1/kernel,pi/dense_1/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
АА*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(
Ж
pi/dense_1/kernel/readIdentitypi/dense_1/kernel*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:
АА
Ф
!pi/dense_1/bias/Initializer/zerosConst*
valueBА*    *
_output_shapes	
:А*"
_class
loc:@pi/dense_1/bias*
dtype0
б
pi/dense_1/bias
VariableV2*
shared_name *
shape:А*
dtype0*
_output_shapes	
:А*
	container *"
_class
loc:@pi/dense_1/bias
╟
pi/dense_1/bias/AssignAssignpi/dense_1/bias!pi/dense_1/bias/Initializer/zeros*
T0*
_output_shapes	
:А*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(
{
pi/dense_1/bias/readIdentitypi/dense_1/bias*
_output_shapes	
:А*"
_class
loc:@pi/dense_1/bias*
T0
Ы
pi/dense_1/MatMulMatMulpi/dense/Tanhpi/dense_1/kernel/read*
transpose_b( *
transpose_a( *(
_output_shapes
:         А*
T0
Р
pi/dense_1/BiasAddBiasAddpi/dense_1/MatMulpi/dense_1/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:         А
^
pi/dense_1/TanhTanhpi/dense_1/BiasAdd*(
_output_shapes
:         А*
T0
й
2pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:*$
_class
loc:@pi/dense_2/kernel
Ы
0pi/dense_2/kernel/Initializer/random_uniform/minConst*$
_class
loc:@pi/dense_2/kernel*
valueB
 *Ц(╛*
dtype0*
_output_shapes
: 
Ы
0pi/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ц(>*$
_class
loc:@pi/dense_2/kernel*
dtype0*
_output_shapes
: 
ї
:pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_2/kernel/Initializer/random_uniform/shape*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	А*
T0*
seed2.*
dtype0*

seed
т
0pi/dense_2/kernel/Initializer/random_uniform/subSub0pi/dense_2/kernel/Initializer/random_uniform/max0pi/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
: 
ї
0pi/dense_2/kernel/Initializer/random_uniform/mulMul:pi/dense_2/kernel/Initializer/random_uniform/RandomUniform0pi/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	А*
T0*$
_class
loc:@pi/dense_2/kernel
ч
,pi/dense_2/kernel/Initializer/random_uniformAdd0pi/dense_2/kernel/Initializer/random_uniform/mul0pi/dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
:	А*$
_class
loc:@pi/dense_2/kernel
н
pi/dense_2/kernel
VariableV2*
_output_shapes
:	А*
	container *
shape:	А*
shared_name *$
_class
loc:@pi/dense_2/kernel*
dtype0
▄
pi/dense_2/kernel/AssignAssignpi/dense_2/kernel,pi/dense_2/kernel/Initializer/random_uniform*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes
:	А
Е
pi/dense_2/kernel/readIdentitypi/dense_2/kernel*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	А*
T0
Т
!pi/dense_2/bias/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
Я
pi/dense_2/bias
VariableV2*
shape:*
dtype0*
shared_name *"
_class
loc:@pi/dense_2/bias*
	container *
_output_shapes
:
╞
pi/dense_2/bias/AssignAssignpi/dense_2/bias!pi/dense_2/bias/Initializer/zeros*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
z
pi/dense_2/bias/readIdentitypi/dense_2/bias*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
Ь
pi/dense_2/MatMulMatMulpi/dense_1/Tanhpi/dense_2/kernel/read*
transpose_a( *'
_output_shapes
:         *
T0*
transpose_b( 
П
pi/dense_2/BiasAddBiasAddpi/dense_2/MatMulpi/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:         *
T0
i
pi/log_std/initial_valueConst*
dtype0*
_output_shapes
:*
valueB"   ┐   ┐
v

pi/log_std
VariableV2*
	container *
dtype0*
shared_name *
_output_shapes
:*
shape:
о
pi/log_std/AssignAssign
pi/log_stdpi/log_std/initial_value*
_output_shapes
:*
_class
loc:@pi/log_std*
use_locking(*
validate_shape(*
T0
k
pi/log_std/readIdentity
pi/log_std*
_output_shapes
:*
T0*
_class
loc:@pi/log_std
C
pi/ExpExppi/log_std/read*
_output_shapes
:*
T0
Z
pi/ShapeShapepi/dense_2/BiasAdd*
out_type0*
T0*
_output_shapes
:
Z
pi/random_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
\
pi/random_normal/stddevConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
Я
%pi/random_normal/RandomStandardNormalRandomStandardNormalpi/Shape*'
_output_shapes
:         *
T0*
dtype0*

seed*
seed2C
Н
pi/random_normal/mulMul%pi/random_normal/RandomStandardNormalpi/random_normal/stddev*'
_output_shapes
:         *
T0
v
pi/random_normalAddpi/random_normal/mulpi/random_normal/mean*
T0*'
_output_shapes
:         
Y
pi/mulMulpi/random_normalpi/Exp*'
_output_shapes
:         *
T0
]
pi/addAddV2pi/dense_2/BiasAddpi/mul*'
_output_shapes
:         *
T0
b
pi/subSubPlaceholder_1pi/dense_2/BiasAdd*
T0*'
_output_shapes
:         
E
pi/Exp_1Exppi/log_std/read*
T0*
_output_shapes
:
O

pi/add_1/yConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
L
pi/add_1AddV2pi/Exp_1
pi/add_1/y*
T0*
_output_shapes
:
Y

pi/truedivRealDivpi/subpi/add_1*
T0*'
_output_shapes
:         
M
pi/pow/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
U
pi/powPow
pi/truedivpi/pow/y*'
_output_shapes
:         *
T0
O

pi/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
Q
pi/mul_1Mul
pi/mul_1/xpi/log_std/read*
_output_shapes
:*
T0
U
pi/add_2AddV2pi/powpi/mul_1*
T0*'
_output_shapes
:         
O

pi/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *О?ы?
Y
pi/add_3AddV2pi/add_2
pi/add_3/y*
T0*'
_output_shapes
:         
O

pi/mul_2/xConst*
valueB
 *   ┐*
dtype0*
_output_shapes
: 
W
pi/mul_2Mul
pi/mul_2/xpi/add_3*
T0*'
_output_shapes
:         
Z
pi/Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
|
pi/SumSumpi/mul_2pi/Sum/reduction_indices*
	keep_dims( *#
_output_shapes
:         *
T0*

Tidx0
]
pi/sub_1Subpi/addpi/dense_2/BiasAdd*
T0*'
_output_shapes
:         
E
pi/Exp_2Exppi/log_std/read*
_output_shapes
:*
T0
O

pi/add_4/yConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
L
pi/add_4AddV2pi/Exp_2
pi/add_4/y*
T0*
_output_shapes
:
]
pi/truediv_1RealDivpi/sub_1pi/add_4*'
_output_shapes
:         *
T0
O

pi/pow_1/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0
[
pi/pow_1Powpi/truediv_1
pi/pow_1/y*'
_output_shapes
:         *
T0
O

pi/mul_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *   @
Q
pi/mul_3Mul
pi/mul_3/xpi/log_std/read*
_output_shapes
:*
T0
W
pi/add_5AddV2pi/pow_1pi/mul_3*
T0*'
_output_shapes
:         
O

pi/add_6/yConst*
dtype0*
_output_shapes
: *
valueB
 *О?ы?
Y
pi/add_6AddV2pi/add_5
pi/add_6/y*'
_output_shapes
:         *
T0
O

pi/mul_4/xConst*
dtype0*
valueB
 *   ┐*
_output_shapes
: 
W
pi/mul_4Mul
pi/mul_4/xpi/add_6*
T0*'
_output_shapes
:         
\
pi/Sum_1/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0
А
pi/Sum_1Sumpi/mul_4pi/Sum_1/reduction_indices*

Tidx0*#
_output_shapes
:         *
T0*
	keep_dims( 
q
pi/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:         *
shape:         
s
pi/Placeholder_1Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
O

pi/mul_5/xConst*
valueB
 *   @*
_output_shapes
: *
dtype0
Q
pi/mul_5Mul
pi/mul_5/xpi/log_std/read*
_output_shapes
:*
T0
>
pi/Exp_3Exppi/mul_5*
T0*
_output_shapes
:
O

pi/mul_6/xConst*
valueB
 *   @*
_output_shapes
: *
dtype0
_
pi/mul_6Mul
pi/mul_6/xpi/Placeholder_1*'
_output_shapes
:         *
T0
K
pi/Exp_4Exppi/mul_6*
T0*'
_output_shapes
:         
e
pi/sub_2Subpi/Placeholderpi/dense_2/BiasAdd*
T0*'
_output_shapes
:         
O

pi/pow_2/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
W
pi/pow_2Powpi/sub_2
pi/pow_2/y*
T0*'
_output_shapes
:         
W
pi/add_7AddV2pi/pow_2pi/Exp_3*'
_output_shapes
:         *
T0
O

pi/add_8/yConst*
_output_shapes
: *
valueB
 *w╠+2*
dtype0
Y
pi/add_8AddV2pi/Exp_4
pi/add_8/y*'
_output_shapes
:         *
T0
]
pi/truediv_2RealDivpi/add_7pi/add_8*'
_output_shapes
:         *
T0
O

pi/sub_3/yConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
[
pi/sub_3Subpi/truediv_2
pi/sub_3/y*
T0*'
_output_shapes
:         
O

pi/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
W
pi/mul_7Mul
pi/mul_7/xpi/sub_3*
T0*'
_output_shapes
:         
_
pi/add_9AddV2pi/mul_7pi/Placeholder_1*
T0*'
_output_shapes
:         
\
pi/sub_4Subpi/add_9pi/log_std/read*
T0*'
_output_shapes
:         
\
pi/Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
А
pi/Sum_2Sumpi/sub_4pi/Sum_2/reduction_indices*

Tidx0*#
_output_shapes
:         *
	keep_dims( *
T0
R
pi/ConstConst*
valueB: *
_output_shapes
:*
dtype0
a
pi/MeanMeanpi/Sum_2pi/Const*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
P
pi/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *╟Я╡?
U
	pi/add_10AddV2pi/log_std/readpi/add_10/y*
T0*
_output_shapes
:
e
pi/Sum_3/reduction_indicesConst*
valueB :
         *
dtype0*
_output_shapes
: 
t
pi/Sum_3Sum	pi/add_10pi/Sum_3/reduction_indices*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
M

pi/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
e
	pi/Mean_1Meanpi/Sum_3
pi/Const_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
е
0vf/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"<      *"
_class
loc:@vf/dense/kernel
Ч
.vf/dense/kernel/Initializer/random_uniform/minConst*"
_class
loc:@vf/dense/kernel*
valueB
 *╛*
_output_shapes
: *
dtype0
Ч
.vf/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*"
_class
loc:@vf/dense/kernel*
_output_shapes
: *
valueB
 *>
Ё
8vf/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vf/dense/kernel/Initializer/random_uniform/shape*

seed*
dtype0*
_output_shapes
:	<А*"
_class
loc:@vf/dense/kernel*
T0*
seed2К
┌
.vf/dense/kernel/Initializer/random_uniform/subSub.vf/dense/kernel/Initializer/random_uniform/max.vf/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
: 
э
.vf/dense/kernel/Initializer/random_uniform/mulMul8vf/dense/kernel/Initializer/random_uniform/RandomUniform.vf/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	<А*
T0*"
_class
loc:@vf/dense/kernel
▀
*vf/dense/kernel/Initializer/random_uniformAdd.vf/dense/kernel/Initializer/random_uniform/mul.vf/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	<А*
T0*"
_class
loc:@vf/dense/kernel
й
vf/dense/kernel
VariableV2*
	container *
shared_name *"
_class
loc:@vf/dense/kernel*
shape:	<А*
_output_shapes
:	<А*
dtype0
╘
vf/dense/kernel/AssignAssignvf/dense/kernel*vf/dense/kernel/Initializer/random_uniform*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<А*
validate_shape(*
T0

vf/dense/kernel/readIdentityvf/dense/kernel*
_output_shapes
:	<А*
T0*"
_class
loc:@vf/dense/kernel
Р
vf/dense/bias/Initializer/zerosConst* 
_class
loc:@vf/dense/bias*
dtype0*
valueBА*    *
_output_shapes	
:А
Э
vf/dense/bias
VariableV2*
	container *
shared_name *
shape:А* 
_class
loc:@vf/dense/bias*
dtype0*
_output_shapes	
:А
┐
vf/dense/bias/AssignAssignvf/dense/biasvf/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
T0*
use_locking(* 
_class
loc:@vf/dense/bias
u
vf/dense/bias/readIdentityvf/dense/bias* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:А
Х
vf/dense/MatMulMatMulPlaceholdervf/dense/kernel/read*
T0*
transpose_a( *
transpose_b( *(
_output_shapes
:         А
К
vf/dense/BiasAddBiasAddvf/dense/MatMulvf/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
Z
vf/dense/TanhTanhvf/dense/BiasAdd*
T0*(
_output_shapes
:         А
й
2vf/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*$
_class
loc:@vf/dense_1/kernel*
valueB"      
Ы
0vf/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *╫│▌╜*$
_class
loc:@vf/dense_1/kernel
Ы
0vf/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *╫│▌=*$
_class
loc:@vf/dense_1/kernel*
_output_shapes
: 
ў
:vf/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_1/kernel/Initializer/random_uniform/shape*$
_class
loc:@vf/dense_1/kernel*
seed2Ы*

seed* 
_output_shapes
:
АА*
T0*
dtype0
т
0vf/dense_1/kernel/Initializer/random_uniform/subSub0vf/dense_1/kernel/Initializer/random_uniform/max0vf/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@vf/dense_1/kernel
Ў
0vf/dense_1/kernel/Initializer/random_uniform/mulMul:vf/dense_1/kernel/Initializer/random_uniform/RandomUniform0vf/dense_1/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АА*
T0*$
_class
loc:@vf/dense_1/kernel
ш
,vf/dense_1/kernel/Initializer/random_uniformAdd0vf/dense_1/kernel/Initializer/random_uniform/mul0vf/dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
АА*
T0*$
_class
loc:@vf/dense_1/kernel
п
vf/dense_1/kernel
VariableV2*$
_class
loc:@vf/dense_1/kernel*
	container *
dtype0* 
_output_shapes
:
АА*
shape:
АА*
shared_name 
▌
vf/dense_1/kernel/AssignAssignvf/dense_1/kernel,vf/dense_1/kernel/Initializer/random_uniform*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:
АА
Ж
vf/dense_1/kernel/readIdentityvf/dense_1/kernel*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
АА
Ф
!vf/dense_1/bias/Initializer/zerosConst*
_output_shapes	
:А*
valueBА*    *"
_class
loc:@vf/dense_1/bias*
dtype0
б
vf/dense_1/bias
VariableV2*"
_class
loc:@vf/dense_1/bias*
dtype0*
shared_name *
	container *
shape:А*
_output_shapes	
:А
╟
vf/dense_1/bias/AssignAssignvf/dense_1/bias!vf/dense_1/bias/Initializer/zeros*
validate_shape(*
T0*
_output_shapes	
:А*
use_locking(*"
_class
loc:@vf/dense_1/bias
{
vf/dense_1/bias/readIdentityvf/dense_1/bias*
T0*
_output_shapes	
:А*"
_class
loc:@vf/dense_1/bias
Ы
vf/dense_1/MatMulMatMulvf/dense/Tanhvf/dense_1/kernel/read*
transpose_b( *
transpose_a( *(
_output_shapes
:         А*
T0
Р
vf/dense_1/BiasAddBiasAddvf/dense_1/MatMulvf/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
^
vf/dense_1/TanhTanhvf/dense_1/BiasAdd*(
_output_shapes
:         А*
T0
й
2vf/dense_2/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@vf/dense_2/kernel*
dtype0*
valueB"      *
_output_shapes
:
Ы
0vf/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*$
_class
loc:@vf/dense_2/kernel*
valueB
 *Iv╛*
_output_shapes
: 
Ы
0vf/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *Iv>*
_output_shapes
: *
dtype0*$
_class
loc:@vf/dense_2/kernel
Ў
:vf/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_2/kernel/Initializer/random_uniform/shape*
_output_shapes
:	А*
seed2м*$
_class
loc:@vf/dense_2/kernel*
dtype0*
T0*

seed
т
0vf/dense_2/kernel/Initializer/random_uniform/subSub0vf/dense_2/kernel/Initializer/random_uniform/max0vf/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@vf/dense_2/kernel
ї
0vf/dense_2/kernel/Initializer/random_uniform/mulMul:vf/dense_2/kernel/Initializer/random_uniform/RandomUniform0vf/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	А*$
_class
loc:@vf/dense_2/kernel*
T0
ч
,vf/dense_2/kernel/Initializer/random_uniformAdd0vf/dense_2/kernel/Initializer/random_uniform/mul0vf/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
:	А*$
_class
loc:@vf/dense_2/kernel*
T0
н
vf/dense_2/kernel
VariableV2*
shared_name *$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	А*
	container *
dtype0*
shape:	А
▄
vf/dense_2/kernel/AssignAssignvf/dense_2/kernel,vf/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	А*
use_locking(
Е
vf/dense_2/kernel/readIdentityvf/dense_2/kernel*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	А
Т
!vf/dense_2/bias/Initializer/zerosConst*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
valueB*    *
dtype0
Я
vf/dense_2/bias
VariableV2*
dtype0*"
_class
loc:@vf/dense_2/bias*
	container *
shape:*
shared_name *
_output_shapes
:
╞
vf/dense_2/bias/AssignAssignvf/dense_2/bias!vf/dense_2/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
z
vf/dense_2/bias/readIdentityvf/dense_2/bias*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0
Ь
vf/dense_2/MatMulMatMulvf/dense_1/Tanhvf/dense_2/kernel/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
П
vf/dense_2/BiasAddBiasAddvf/dense_2/MatMulvf/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
n

vf/SqueezeSqueezevf/dense_2/BiasAdd*
squeeze_dims
*#
_output_shapes
:         *
T0
е
0vc/dense/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@vc/dense/kernel*
_output_shapes
:*
valueB"<      *
dtype0
Ч
.vc/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *╛*"
_class
loc:@vc/dense/kernel
Ч
.vc/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *>*"
_class
loc:@vc/dense/kernel*
_output_shapes
: 
Ё
8vc/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vc/dense/kernel/Initializer/random_uniform/shape*
dtype0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<А*
T0*

seed*
seed2╜
┌
.vc/dense/kernel/Initializer/random_uniform/subSub.vc/dense/kernel/Initializer/random_uniform/max.vc/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@vc/dense/kernel*
_output_shapes
: *
T0
э
.vc/dense/kernel/Initializer/random_uniform/mulMul8vc/dense/kernel/Initializer/random_uniform/RandomUniform.vc/dense/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<А
▀
*vc/dense/kernel/Initializer/random_uniformAdd.vc/dense/kernel/Initializer/random_uniform/mul.vc/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<А
й
vc/dense/kernel
VariableV2*
_output_shapes
:	<А*
dtype0*
	container *
shared_name *"
_class
loc:@vc/dense/kernel*
shape:	<А
╘
vc/dense/kernel/AssignAssignvc/dense/kernel*vc/dense/kernel/Initializer/random_uniform*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	<А

vc/dense/kernel/readIdentityvc/dense/kernel*
T0*
_output_shapes
:	<А*"
_class
loc:@vc/dense/kernel
Р
vc/dense/bias/Initializer/zerosConst* 
_class
loc:@vc/dense/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Э
vc/dense/bias
VariableV2*
shared_name *
shape:А*
_output_shapes	
:А* 
_class
loc:@vc/dense/bias*
	container *
dtype0
┐
vc/dense/bias/AssignAssignvc/dense/biasvc/dense/bias/Initializer/zeros* 
_class
loc:@vc/dense/bias*
_output_shapes	
:А*
T0*
validate_shape(*
use_locking(
u
vc/dense/bias/readIdentityvc/dense/bias*
_output_shapes	
:А* 
_class
loc:@vc/dense/bias*
T0
Х
vc/dense/MatMulMatMulPlaceholdervc/dense/kernel/read*
transpose_a( *
T0*(
_output_shapes
:         А*
transpose_b( 
К
vc/dense/BiasAddBiasAddvc/dense/MatMulvc/dense/bias/read*
T0*(
_output_shapes
:         А*
data_formatNHWC
Z
vc/dense/TanhTanhvc/dense/BiasAdd*(
_output_shapes
:         А*
T0
й
2vc/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:*$
_class
loc:@vc/dense_1/kernel
Ы
0vc/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *╫│▌╜*$
_class
loc:@vc/dense_1/kernel*
dtype0
Ы
0vc/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *╫│▌=*
_output_shapes
: *$
_class
loc:@vc/dense_1/kernel
ў
:vc/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_1/kernel/Initializer/random_uniform/shape*$
_class
loc:@vc/dense_1/kernel*
seed2╬*
T0*

seed* 
_output_shapes
:
АА*
dtype0
т
0vc/dense_1/kernel/Initializer/random_uniform/subSub0vc/dense_1/kernel/Initializer/random_uniform/max0vc/dense_1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@vc/dense_1/kernel*
_output_shapes
: 
Ў
0vc/dense_1/kernel/Initializer/random_uniform/mulMul:vc/dense_1/kernel/Initializer/random_uniform/RandomUniform0vc/dense_1/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АА*
T0*$
_class
loc:@vc/dense_1/kernel
ш
,vc/dense_1/kernel/Initializer/random_uniformAdd0vc/dense_1/kernel/Initializer/random_uniform/mul0vc/dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
АА*$
_class
loc:@vc/dense_1/kernel*
T0
п
vc/dense_1/kernel
VariableV2*
	container * 
_output_shapes
:
АА*
dtype0*$
_class
loc:@vc/dense_1/kernel*
shape:
АА*
shared_name 
▌
vc/dense_1/kernel/AssignAssignvc/dense_1/kernel,vc/dense_1/kernel/Initializer/random_uniform*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:
АА*
validate_shape(
Ж
vc/dense_1/kernel/readIdentityvc/dense_1/kernel* 
_output_shapes
:
АА*
T0*$
_class
loc:@vc/dense_1/kernel
Ф
!vc/dense_1/bias/Initializer/zerosConst*
valueBА*    *
dtype0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:А
б
vc/dense_1/bias
VariableV2*
_output_shapes	
:А*
shared_name *"
_class
loc:@vc/dense_1/bias*
	container *
shape:А*
dtype0
╟
vc/dense_1/bias/AssignAssignvc/dense_1/bias!vc/dense_1/bias/Initializer/zeros*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:А*
validate_shape(
{
vc/dense_1/bias/readIdentityvc/dense_1/bias*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:А*
T0
Ы
vc/dense_1/MatMulMatMulvc/dense/Tanhvc/dense_1/kernel/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:         А
Р
vc/dense_1/BiasAddBiasAddvc/dense_1/MatMulvc/dense_1/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:         А
^
vc/dense_1/TanhTanhvc/dense_1/BiasAdd*(
_output_shapes
:         А*
T0
й
2vc/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:*
valueB"      
Ы
0vc/dense_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*$
_class
loc:@vc/dense_2/kernel*
valueB
 *Iv╛
Ы
0vc/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *Iv>*
_output_shapes
: *
dtype0*$
_class
loc:@vc/dense_2/kernel
Ў
:vc/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	А*
seed2▀*
T0*

seed
т
0vc/dense_2/kernel/Initializer/random_uniform/subSub0vc/dense_2/kernel/Initializer/random_uniform/max0vc/dense_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
: 
ї
0vc/dense_2/kernel/Initializer/random_uniform/mulMul:vc/dense_2/kernel/Initializer/random_uniform/RandomUniform0vc/dense_2/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel
ч
,vc/dense_2/kernel/Initializer/random_uniformAdd0vc/dense_2/kernel/Initializer/random_uniform/mul0vc/dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel
н
vc/dense_2/kernel
VariableV2*
	container *
_output_shapes
:	А*
dtype0*
shared_name *
shape:	А*$
_class
loc:@vc/dense_2/kernel
▄
vc/dense_2/kernel/AssignAssignvc/dense_2/kernel,vc/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel*
T0*
use_locking(
Е
vc/dense_2/kernel/readIdentityvc/dense_2/kernel*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	А
Т
!vc/dense_2/bias/Initializer/zerosConst*
_output_shapes
:*
dtype0*"
_class
loc:@vc/dense_2/bias*
valueB*    
Я
vc/dense_2/bias
VariableV2*
_output_shapes
:*
dtype0*
	container *
shared_name *"
_class
loc:@vc/dense_2/bias*
shape:
╞
vc/dense_2/bias/AssignAssignvc/dense_2/bias!vc/dense_2/bias/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias
z
vc/dense_2/bias/readIdentityvc/dense_2/bias*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
Ь
vc/dense_2/MatMulMatMulvc/dense_1/Tanhvc/dense_2/kernel/read*
transpose_b( *
transpose_a( *'
_output_shapes
:         *
T0
П
vc/dense_2/BiasAddBiasAddvc/dense_2/MatMulvc/dense_2/bias/read*
T0*'
_output_shapes
:         *
data_formatNHWC
n

vc/SqueezeSqueezevc/dense_2/BiasAdd*
T0*
squeeze_dims
*#
_output_shapes
:         
@
NegNegpi/Sum*#
_output_shapes
:         *
T0
O
ConstConst*
valueB: *
_output_shapes
:*
dtype0
V
MeanMeanNegConst*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
O
subSubpi/SumPlaceholder_6*
T0*#
_output_shapes
:         
=
ExpExpsub*
T0*#
_output_shapes
:         
L
mulMulExpPlaceholder_2*
T0*#
_output_shapes
:         
Q
Const_1Const*
_output_shapes
:*
valueB: *
dtype0
Z
Mean_1MeanmulConst_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
N
mul_1MulExpPlaceholder_3*
T0*#
_output_shapes
:         
Q
Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
\
Mean_2Meanmul_1Const_2*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
L
mul_2/xConst*
_output_shapes
: *
valueB
 *    *
dtype0
A
mul_2Mulmul_2/x	pi/Mean_1*
_output_shapes
: *
T0
<
addAddV2Mean_1mul_2*
_output_shapes
: *
T0
2
Neg_1Negadd*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*

index_type0*
T0*
_output_shapes
: 
P
gradients/Neg_1_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
m
#gradients/Mean_1_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
Ъ
gradients/Mean_1_grad/ReshapeReshapegradients/Neg_1_grad/Neg#gradients/Mean_1_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
^
gradients/Mean_1_grad/ShapeShapemul*
T0*
_output_shapes
:*
out_type0
Ю
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:         
`
gradients/Mean_1_grad/Shape_1Shapemul*
_output_shapes
:*
T0*
out_type0
`
gradients/Mean_1_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
e
gradients/Mean_1_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ь
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_1gradients/Mean_1_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
g
gradients/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
а
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
a
gradients/Mean_1_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
И
gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
Ж
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
В
gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*
_output_shapes
: *

DstT0*
Truncate( *

SrcT0
О
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*#
_output_shapes
:         *
T0
e
gradients/mul_2_grad/MulMulgradients/Neg_1_grad/Neg	pi/Mean_1*
T0*
_output_shapes
: 
e
gradients/mul_2_grad/Mul_1Mulgradients/Neg_1_grad/Negmul_2/x*
_output_shapes
: *
T0
[
gradients/mul_grad/ShapeShapeExp*
T0*
out_type0*
_output_shapes
:
g
gradients/mul_grad/Shape_1ShapePlaceholder_2*
T0*
out_type0*
_output_shapes
:
┤
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
y
gradients/mul_grad/MulMulgradients/Mean_1_grad/truedivPlaceholder_2*#
_output_shapes
:         *
T0
Я
gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
У
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*#
_output_shapes
:         *
Tshape0
q
gradients/mul_grad/Mul_1MulExpgradients/Mean_1_grad/truediv*
T0*#
_output_shapes
:         
е
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
Щ
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*#
_output_shapes
:         *
T0*
Tshape0
i
&gradients/pi/Mean_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Ю
 gradients/pi/Mean_1_grad/ReshapeReshapegradients/mul_2_grad/Mul_1&gradients/pi/Mean_1_grad/Reshape/shape*
_output_shapes
: *
T0*
Tshape0
a
gradients/pi/Mean_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
Ъ
gradients/pi/Mean_1_grad/TileTile gradients/pi/Mean_1_grad/Reshapegradients/pi/Mean_1_grad/Const*
T0*

Tmultiples0*
_output_shapes
: 
e
 gradients/pi/Mean_1_grad/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  А?
Н
 gradients/pi/Mean_1_grad/truedivRealDivgradients/pi/Mean_1_grad/Tile gradients/pi/Mean_1_grad/Const_1*
T0*
_output_shapes
: 
l
gradients/Exp_grad/mulMulgradients/mul_grad/ReshapeExp*#
_output_shapes
:         *
T0
h
gradients/pi/Sum_3_grad/Cast/xConst*
_output_shapes
:*
dtype0*
valueB:
s
 gradients/pi/Sum_3_grad/Cast_1/xConst*
valueB:
         *
dtype0*
_output_shapes
:
^
gradients/pi/Sum_3_grad/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
Й
gradients/pi/Sum_3_grad/addAddV2 gradients/pi/Sum_3_grad/Cast_1/xgradients/pi/Sum_3_grad/Size*
T0*
_output_shapes
:
З
gradients/pi/Sum_3_grad/modFloorModgradients/pi/Sum_3_grad/addgradients/pi/Sum_3_grad/Size*
_output_shapes
:*
T0
g
gradients/pi/Sum_3_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
e
#gradients/pi/Sum_3_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
e
#gradients/pi/Sum_3_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
╢
gradients/pi/Sum_3_grad/rangeRange#gradients/pi/Sum_3_grad/range/startgradients/pi/Sum_3_grad/Size#gradients/pi/Sum_3_grad/range/delta*

Tidx0*
_output_shapes
:
d
"gradients/pi/Sum_3_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Ю
gradients/pi/Sum_3_grad/FillFillgradients/pi/Sum_3_grad/Shape"gradients/pi/Sum_3_grad/Fill/value*

index_type0*
_output_shapes
:*
T0
▐
%gradients/pi/Sum_3_grad/DynamicStitchDynamicStitchgradients/pi/Sum_3_grad/rangegradients/pi/Sum_3_grad/modgradients/pi/Sum_3_grad/Cast/xgradients/pi/Sum_3_grad/Fill*
N*
_output_shapes
:*
T0
k
!gradients/pi/Sum_3_grad/Maximum/xConst*
_output_shapes
:*
dtype0*
valueB:
c
!gradients/pi/Sum_3_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
Х
gradients/pi/Sum_3_grad/MaximumMaximum!gradients/pi/Sum_3_grad/Maximum/x!gradients/pi/Sum_3_grad/Maximum/y*
T0*
_output_shapes
:
l
"gradients/pi/Sum_3_grad/floordiv/xConst*
valueB:*
dtype0*
_output_shapes
:
Ц
 gradients/pi/Sum_3_grad/floordivFloorDiv"gradients/pi/Sum_3_grad/floordiv/xgradients/pi/Sum_3_grad/Maximum*
_output_shapes
:*
T0
o
%gradients/pi/Sum_3_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
ж
gradients/pi/Sum_3_grad/ReshapeReshape gradients/pi/Mean_1_grad/truediv%gradients/pi/Sum_3_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
p
&gradients/pi/Sum_3_grad/Tile/multiplesConst*
valueB:*
_output_shapes
:*
dtype0
д
gradients/pi/Sum_3_grad/TileTilegradients/pi/Sum_3_grad/Reshape&gradients/pi/Sum_3_grad/Tile/multiples*
T0*
_output_shapes
:*

Tmultiples0
^
gradients/sub_grad/ShapeShapepi/Sum*
out_type0*
T0*
_output_shapes
:
g
gradients/sub_grad/Shape_1ShapePlaceholder_6*
T0*
_output_shapes
:*
out_type0
┤
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Я
gradients/sub_grad/SumSumgradients/Exp_grad/mul(gradients/sub_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
У
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*#
_output_shapes
:         *
T0*
Tshape0
c
gradients/sub_grad/NegNeggradients/Exp_grad/mul*
T0*#
_output_shapes
:         
г
gradients/sub_grad/Sum_1Sumgradients/sub_grad/Neg*gradients/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
Щ
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Sum_1gradients/sub_grad/Shape_1*
Tshape0*
T0*#
_output_shapes
:         
{
1gradients/pi/add_10_grad/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*
valueB:
t
1gradients/pi/add_10_grad/BroadcastGradientArgs/s1Const*
dtype0*
valueB *
_output_shapes
: 
ъ
.gradients/pi/add_10_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/pi/add_10_grad/BroadcastGradientArgs/s01gradients/pi/add_10_grad/BroadcastGradientArgs/s1*2
_output_shapes 
:         :         *
T0
x
.gradients/pi/add_10_grad/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
п
gradients/pi/add_10_grad/SumSumgradients/pi/Sum_3_grad/Tile.gradients/pi/add_10_grad/Sum/reduction_indices*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
i
&gradients/pi/add_10_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0
а
 gradients/pi/add_10_grad/ReshapeReshapegradients/pi/add_10_grad/Sum&gradients/pi/add_10_grad/Reshape/shape*
Tshape0*
_output_shapes
: *
T0
c
gradients/pi/Sum_grad/ShapeShapepi/mul_2*
T0*
_output_shapes
:*
out_type0
М
gradients/pi/Sum_grad/SizeConst*
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
value	B :*
_output_shapes
: 
й
gradients/pi/Sum_grad/addAddV2pi/Sum/reduction_indicesgradients/pi/Sum_grad/Size*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
н
gradients/pi/Sum_grad/modFloorModgradients/pi/Sum_grad/addgradients/pi/Sum_grad/Size*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
T0
Р
gradients/pi/Sum_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
У
!gradients/pi/Sum_grad/range/startConst*
value	B : *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
dtype0
У
!gradients/pi/Sum_grad/range/deltaConst*
value	B :*
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: 
▐
gradients/pi/Sum_grad/rangeRange!gradients/pi/Sum_grad/range/startgradients/pi/Sum_grad/Size!gradients/pi/Sum_grad/range/delta*

Tidx0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:
Т
 gradients/pi/Sum_grad/Fill/valueConst*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: *
value	B :
╞
gradients/pi/Sum_grad/FillFillgradients/pi/Sum_grad/Shape_1 gradients/pi/Sum_grad/Fill/value*

index_type0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
T0
Г
#gradients/pi/Sum_grad/DynamicStitchDynamicStitchgradients/pi/Sum_grad/rangegradients/pi/Sum_grad/modgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Fill*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:*
N
С
gradients/pi/Sum_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
├
gradients/pi/Sum_grad/MaximumMaximum#gradients/pi/Sum_grad/DynamicStitchgradients/pi/Sum_grad/Maximum/y*
_output_shapes
:*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
╗
gradients/pi/Sum_grad/floordivFloorDivgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Maximum*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:*
T0
▓
gradients/pi/Sum_grad/ReshapeReshapegradients/sub_grad/Reshape#gradients/pi/Sum_grad/DynamicStitch*0
_output_shapes
:                  *
Tshape0*
T0
е
gradients/pi/Sum_grad/TileTilegradients/pi/Sum_grad/Reshapegradients/pi/Sum_grad/floordiv*
T0*

Tmultiples0*'
_output_shapes
:         
e
gradients/pi/mul_2_grad/ShapeShape
pi/mul_2/x*
out_type0*
T0*
_output_shapes
: 
g
gradients/pi/mul_2_grad/Shape_1Shapepi/add_3*
T0*
out_type0*
_output_shapes
:
├
-gradients/pi/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/mul_2_grad/Shapegradients/pi/mul_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
z
gradients/pi/mul_2_grad/MulMulgradients/pi/Sum_grad/Tilepi/add_3*'
_output_shapes
:         *
T0
о
gradients/pi/mul_2_grad/SumSumgradients/pi/mul_2_grad/Mul-gradients/pi/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Х
gradients/pi/mul_2_grad/ReshapeReshapegradients/pi/mul_2_grad/Sumgradients/pi/mul_2_grad/Shape*
_output_shapes
: *
T0*
Tshape0
~
gradients/pi/mul_2_grad/Mul_1Mul
pi/mul_2/xgradients/pi/Sum_grad/Tile*
T0*'
_output_shapes
:         
┤
gradients/pi/mul_2_grad/Sum_1Sumgradients/pi/mul_2_grad/Mul_1/gradients/pi/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
м
!gradients/pi/mul_2_grad/Reshape_1Reshapegradients/pi/mul_2_grad/Sum_1gradients/pi/mul_2_grad/Shape_1*
Tshape0*'
_output_shapes
:         *
T0
e
gradients/pi/add_3_grad/ShapeShapepi/add_2*
T0*
out_type0*
_output_shapes
:
g
gradients/pi/add_3_grad/Shape_1Shape
pi/add_3/y*
_output_shapes
: *
out_type0*
T0
├
-gradients/pi/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_3_grad/Shapegradients/pi/add_3_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┤
gradients/pi/add_3_grad/SumSum!gradients/pi/mul_2_grad/Reshape_1-gradients/pi/add_3_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
ж
gradients/pi/add_3_grad/ReshapeReshapegradients/pi/add_3_grad/Sumgradients/pi/add_3_grad/Shape*'
_output_shapes
:         *
Tshape0*
T0
╕
gradients/pi/add_3_grad/Sum_1Sum!gradients/pi/mul_2_grad/Reshape_1/gradients/pi/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
Ы
!gradients/pi/add_3_grad/Reshape_1Reshapegradients/pi/add_3_grad/Sum_1gradients/pi/add_3_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
c
gradients/pi/add_2_grad/ShapeShapepi/pow*
T0*
out_type0*
_output_shapes
:
g
gradients/pi/add_2_grad/Shape_1Shapepi/mul_1*
out_type0*
_output_shapes
:*
T0
├
-gradients/pi/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_2_grad/Shapegradients/pi/add_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
▓
gradients/pi/add_2_grad/SumSumgradients/pi/add_3_grad/Reshape-gradients/pi/add_2_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
ж
gradients/pi/add_2_grad/ReshapeReshapegradients/pi/add_2_grad/Sumgradients/pi/add_2_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
╢
gradients/pi/add_2_grad/Sum_1Sumgradients/pi/add_3_grad/Reshape/gradients/pi/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Я
!gradients/pi/add_2_grad/Reshape_1Reshapegradients/pi/add_2_grad/Sum_1gradients/pi/add_2_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
e
gradients/pi/pow_grad/ShapeShape
pi/truediv*
T0*
_output_shapes
:*
out_type0
c
gradients/pi/pow_grad/Shape_1Shapepi/pow/y*
T0*
out_type0*
_output_shapes
: 
╜
+gradients/pi/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/pow_grad/Shapegradients/pi/pow_grad/Shape_1*2
_output_shapes 
:         :         *
T0
}
gradients/pi/pow_grad/mulMulgradients/pi/add_2_grad/Reshapepi/pow/y*
T0*'
_output_shapes
:         
`
gradients/pi/pow_grad/sub/yConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
h
gradients/pi/pow_grad/subSubpi/pow/ygradients/pi/pow_grad/sub/y*
T0*
_output_shapes
: 
y
gradients/pi/pow_grad/PowPow
pi/truedivgradients/pi/pow_grad/sub*'
_output_shapes
:         *
T0
К
gradients/pi/pow_grad/mul_1Mulgradients/pi/pow_grad/mulgradients/pi/pow_grad/Pow*'
_output_shapes
:         *
T0
к
gradients/pi/pow_grad/SumSumgradients/pi/pow_grad/mul_1+gradients/pi/pow_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
а
gradients/pi/pow_grad/ReshapeReshapegradients/pi/pow_grad/Sumgradients/pi/pow_grad/Shape*
Tshape0*
T0*'
_output_shapes
:         
d
gradients/pi/pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
З
gradients/pi/pow_grad/GreaterGreater
pi/truedivgradients/pi/pow_grad/Greater/y*'
_output_shapes
:         *
T0
o
%gradients/pi/pow_grad/ones_like/ShapeShape
pi/truediv*
T0*
out_type0*
_output_shapes
:
j
%gradients/pi/pow_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
╣
gradients/pi/pow_grad/ones_likeFill%gradients/pi/pow_grad/ones_like/Shape%gradients/pi/pow_grad/ones_like/Const*

index_type0*'
_output_shapes
:         *
T0
д
gradients/pi/pow_grad/SelectSelectgradients/pi/pow_grad/Greater
pi/truedivgradients/pi/pow_grad/ones_like*
T0*'
_output_shapes
:         
p
gradients/pi/pow_grad/LogLoggradients/pi/pow_grad/Select*
T0*'
_output_shapes
:         
k
 gradients/pi/pow_grad/zeros_like	ZerosLike
pi/truediv*
T0*'
_output_shapes
:         
╢
gradients/pi/pow_grad/Select_1Selectgradients/pi/pow_grad/Greatergradients/pi/pow_grad/Log gradients/pi/pow_grad/zeros_like*'
_output_shapes
:         *
T0
}
gradients/pi/pow_grad/mul_2Mulgradients/pi/add_2_grad/Reshapepi/pow*'
_output_shapes
:         *
T0
С
gradients/pi/pow_grad/mul_3Mulgradients/pi/pow_grad/mul_2gradients/pi/pow_grad/Select_1*
T0*'
_output_shapes
:         
о
gradients/pi/pow_grad/Sum_1Sumgradients/pi/pow_grad/mul_3-gradients/pi/pow_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
Х
gradients/pi/pow_grad/Reshape_1Reshapegradients/pi/pow_grad/Sum_1gradients/pi/pow_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
s
0gradients/pi/mul_1_grad/BroadcastGradientArgs/s0Const*
_output_shapes
: *
valueB *
dtype0
z
0gradients/pi/mul_1_grad/BroadcastGradientArgs/s1Const*
dtype0*
_output_shapes
:*
valueB:
ч
-gradients/pi/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients/pi/mul_1_grad/BroadcastGradientArgs/s00gradients/pi/mul_1_grad/BroadcastGradientArgs/s1*
T0*2
_output_shapes 
:         :         
{
gradients/pi/mul_1_grad/MulMul!gradients/pi/add_2_grad/Reshape_1pi/log_std/read*
_output_shapes
:*
T0
w
-gradients/pi/mul_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
м
gradients/pi/mul_1_grad/SumSumgradients/pi/mul_1_grad/Mul-gradients/pi/mul_1_grad/Sum/reduction_indices*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
h
%gradients/pi/mul_1_grad/Reshape/shapeConst*
dtype0*
valueB *
_output_shapes
: 
Э
gradients/pi/mul_1_grad/ReshapeReshapegradients/pi/mul_1_grad/Sum%gradients/pi/mul_1_grad/Reshape/shape*
_output_shapes
: *
Tshape0*
T0
x
gradients/pi/mul_1_grad/Mul_1Mul
pi/mul_1/x!gradients/pi/add_2_grad/Reshape_1*
_output_shapes
:*
T0
e
gradients/pi/truediv_grad/ShapeShapepi/sub*
out_type0*
T0*
_output_shapes
:
k
!gradients/pi/truediv_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
╔
/gradients/pi/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/truediv_grad/Shape!gradients/pi/truediv_grad/Shape_1*2
_output_shapes 
:         :         *
T0
З
!gradients/pi/truediv_grad/RealDivRealDivgradients/pi/pow_grad/Reshapepi/add_1*'
_output_shapes
:         *
T0
╕
gradients/pi/truediv_grad/SumSum!gradients/pi/truediv_grad/RealDiv/gradients/pi/truediv_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
м
!gradients/pi/truediv_grad/ReshapeReshapegradients/pi/truediv_grad/Sumgradients/pi/truediv_grad/Shape*'
_output_shapes
:         *
Tshape0*
T0
^
gradients/pi/truediv_grad/NegNegpi/sub*
T0*'
_output_shapes
:         
Й
#gradients/pi/truediv_grad/RealDiv_1RealDivgradients/pi/truediv_grad/Negpi/add_1*'
_output_shapes
:         *
T0
П
#gradients/pi/truediv_grad/RealDiv_2RealDiv#gradients/pi/truediv_grad/RealDiv_1pi/add_1*
T0*'
_output_shapes
:         
Ъ
gradients/pi/truediv_grad/mulMulgradients/pi/pow_grad/Reshape#gradients/pi/truediv_grad/RealDiv_2*'
_output_shapes
:         *
T0
╕
gradients/pi/truediv_grad/Sum_1Sumgradients/pi/truediv_grad/mul1gradients/pi/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
е
#gradients/pi/truediv_grad/Reshape_1Reshapegradients/pi/truediv_grad/Sum_1!gradients/pi/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
h
gradients/pi/sub_grad/ShapeShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
o
gradients/pi/sub_grad/Shape_1Shapepi/dense_2/BiasAdd*
out_type0*
T0*
_output_shapes
:
╜
+gradients/pi/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/sub_grad/Shapegradients/pi/sub_grad/Shape_1*2
_output_shapes 
:         :         *
T0
░
gradients/pi/sub_grad/SumSum!gradients/pi/truediv_grad/Reshape+gradients/pi/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
а
gradients/pi/sub_grad/ReshapeReshapegradients/pi/sub_grad/Sumgradients/pi/sub_grad/Shape*
Tshape0*
T0*'
_output_shapes
:         
u
gradients/pi/sub_grad/NegNeg!gradients/pi/truediv_grad/Reshape*
T0*'
_output_shapes
:         
м
gradients/pi/sub_grad/Sum_1Sumgradients/pi/sub_grad/Neg-gradients/pi/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ж
gradients/pi/sub_grad/Reshape_1Reshapegradients/pi/sub_grad/Sum_1gradients/pi/sub_grad/Shape_1*
T0*'
_output_shapes
:         *
Tshape0
w
-gradients/pi/add_1_grad/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
┤
gradients/pi/add_1_grad/SumSum#gradients/pi/truediv_grad/Reshape_1-gradients/pi/add_1_grad/Sum/reduction_indices*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
h
%gradients/pi/add_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Э
gradients/pi/add_1_grad/ReshapeReshapegradients/pi/add_1_grad/Sum%gradients/pi/add_1_grad/Reshape/shape*
_output_shapes
: *
Tshape0*
T0
Щ
-gradients/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/pi/sub_grad/Reshape_1*
data_formatNHWC*
_output_shapes
:*
T0
v
gradients/pi/Exp_1_grad/mulMul#gradients/pi/truediv_grad/Reshape_1pi/Exp_1*
T0*
_output_shapes
:
├
'gradients/pi/dense_2/MatMul_grad/MatMulMatMulgradients/pi/sub_grad/Reshape_1pi/dense_2/kernel/read*
transpose_b(*(
_output_shapes
:         А*
transpose_a( *
T0
╡
)gradients/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanhgradients/pi/sub_grad/Reshape_1*
transpose_a(*
T0*
transpose_b( *
_output_shapes
:	А
╧
gradients/AddNAddNgradients/pi/Sum_3_grad/Tilegradients/pi/mul_1_grad/Mul_1gradients/pi/Exp_1_grad/mul*
T0*
_output_shapes
:*
N*/
_class%
#!loc:@gradients/pi/Sum_3_grad/Tile
а
'gradients/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh'gradients/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:         А*
T0
в
-gradients/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/pi/dense_1/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:А*
T0
╦
'gradients/pi/dense_1/MatMul_grad/MatMulMatMul'gradients/pi/dense_1/Tanh_grad/TanhGradpi/dense_1/kernel/read*
transpose_a( *
T0*(
_output_shapes
:         А*
transpose_b(
╝
)gradients/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh'gradients/pi/dense_1/Tanh_grad/TanhGrad*
T0* 
_output_shapes
:
АА*
transpose_b( *
transpose_a(
Ь
%gradients/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh'gradients/pi/dense_1/MatMul_grad/MatMul*(
_output_shapes
:         А*
T0
Ю
+gradients/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad%gradients/pi/dense/Tanh_grad/TanhGrad*
T0*
_output_shapes	
:А*
data_formatNHWC
─
%gradients/pi/dense/MatMul_grad/MatMulMatMul%gradients/pi/dense/Tanh_grad/TanhGradpi/dense/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:         <
╡
'gradients/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder%gradients/pi/dense/Tanh_grad/TanhGrad*
transpose_a(*
_output_shapes
:	<А*
T0*
transpose_b( 
`
Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
~
ReshapeReshape'gradients/pi/dense/MatMul_grad/MatMul_1Reshape/shape*
Tshape0*
T0*
_output_shapes	
:Аx
b
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
Ж
	Reshape_1Reshape+gradients/pi/dense/BiasAdd_grad/BiasAddGradReshape_1/shape*
Tshape0*
_output_shapes	
:А*
T0
b
Reshape_2/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
Е
	Reshape_2Reshape)gradients/pi/dense_1/MatMul_grad/MatMul_1Reshape_2/shape*
_output_shapes

:АА*
T0*
Tshape0
b
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
И
	Reshape_3Reshape-gradients/pi/dense_1/BiasAdd_grad/BiasAddGradReshape_3/shape*
T0*
Tshape0*
_output_shapes	
:А
b
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
Д
	Reshape_4Reshape)gradients/pi/dense_2/MatMul_grad/MatMul_1Reshape_4/shape*
_output_shapes	
:А*
Tshape0*
T0
b
Reshape_5/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
З
	Reshape_5Reshape-gradients/pi/dense_2/BiasAdd_grad/BiasAddGradReshape_5/shape*
T0*
_output_shapes
:*
Tshape0
b
Reshape_6/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
h
	Reshape_6Reshapegradients/AddNReshape_6/shape*
_output_shapes
:*
T0*
Tshape0
M
concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
ж
concatConcatV2Reshape	Reshape_1	Reshape_2	Reshape_3	Reshape_4	Reshape_5	Reshape_6concat/axis*
N*

Tidx0*
_output_shapes

:ДА*
T0
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
dtype0*
valueB
 *  А?*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
p
&gradients_1/pi/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Ш
 gradients_1/pi/Mean_grad/ReshapeReshapegradients_1/Fill&gradients_1/pi/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
f
gradients_1/pi/Mean_grad/ShapeShapepi/Sum_2*
out_type0*
T0*
_output_shapes
:
з
gradients_1/pi/Mean_grad/TileTile gradients_1/pi/Mean_grad/Reshapegradients_1/pi/Mean_grad/Shape*

Tmultiples0*#
_output_shapes
:         *
T0
h
 gradients_1/pi/Mean_grad/Shape_1Shapepi/Sum_2*
_output_shapes
:*
T0*
out_type0
c
 gradients_1/pi/Mean_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
h
gradients_1/pi/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
е
gradients_1/pi/Mean_grad/ProdProd gradients_1/pi/Mean_grad/Shape_1gradients_1/pi/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
j
 gradients_1/pi/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
й
gradients_1/pi/Mean_grad/Prod_1Prod gradients_1/pi/Mean_grad/Shape_2 gradients_1/pi/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
d
"gradients_1/pi/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
С
 gradients_1/pi/Mean_grad/MaximumMaximumgradients_1/pi/Mean_grad/Prod_1"gradients_1/pi/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
П
!gradients_1/pi/Mean_grad/floordivFloorDivgradients_1/pi/Mean_grad/Prod gradients_1/pi/Mean_grad/Maximum*
T0*
_output_shapes
: 
И
gradients_1/pi/Mean_grad/CastCast!gradients_1/pi/Mean_grad/floordiv*

DstT0*
_output_shapes
: *
Truncate( *

SrcT0
Ч
 gradients_1/pi/Mean_grad/truedivRealDivgradients_1/pi/Mean_grad/Tilegradients_1/pi/Mean_grad/Cast*
T0*#
_output_shapes
:         
g
gradients_1/pi/Sum_2_grad/ShapeShapepi/sub_4*
T0*
_output_shapes
:*
out_type0
Ф
gradients_1/pi/Sum_2_grad/SizeConst*
dtype0*
value	B :*
_output_shapes
: *2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape
╖
gradients_1/pi/Sum_2_grad/addAddV2pi/Sum_2/reduction_indicesgradients_1/pi/Sum_2_grad/Size*
T0*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
_output_shapes
: 
╜
gradients_1/pi/Sum_2_grad/modFloorModgradients_1/pi/Sum_2_grad/addgradients_1/pi/Sum_2_grad/Size*
_output_shapes
: *
T0*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape
Ш
!gradients_1/pi/Sum_2_grad/Shape_1Const*
valueB *2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
dtype0*
_output_shapes
: 
Ы
%gradients_1/pi/Sum_2_grad/range/startConst*
_output_shapes
: *
dtype0*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
value	B : 
Ы
%gradients_1/pi/Sum_2_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: *2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape
Є
gradients_1/pi/Sum_2_grad/rangeRange%gradients_1/pi/Sum_2_grad/range/startgradients_1/pi/Sum_2_grad/Size%gradients_1/pi/Sum_2_grad/range/delta*
_output_shapes
:*

Tidx0*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape
Ъ
$gradients_1/pi/Sum_2_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: *2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape
╓
gradients_1/pi/Sum_2_grad/FillFill!gradients_1/pi/Sum_2_grad/Shape_1$gradients_1/pi/Sum_2_grad/Fill/value*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
_output_shapes
: *

index_type0*
T0
Ы
'gradients_1/pi/Sum_2_grad/DynamicStitchDynamicStitchgradients_1/pi/Sum_2_grad/rangegradients_1/pi/Sum_2_grad/modgradients_1/pi/Sum_2_grad/Shapegradients_1/pi/Sum_2_grad/Fill*
_output_shapes
:*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
T0*
N
Щ
#gradients_1/pi/Sum_2_grad/Maximum/yConst*
value	B :*
_output_shapes
: *2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
dtype0
╙
!gradients_1/pi/Sum_2_grad/MaximumMaximum'gradients_1/pi/Sum_2_grad/DynamicStitch#gradients_1/pi/Sum_2_grad/Maximum/y*
_output_shapes
:*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
T0
╦
"gradients_1/pi/Sum_2_grad/floordivFloorDivgradients_1/pi/Sum_2_grad/Shape!gradients_1/pi/Sum_2_grad/Maximum*
T0*
_output_shapes
:*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape
└
!gradients_1/pi/Sum_2_grad/ReshapeReshape gradients_1/pi/Mean_grad/truediv'gradients_1/pi/Sum_2_grad/DynamicStitch*
T0*0
_output_shapes
:                  *
Tshape0
▒
gradients_1/pi/Sum_2_grad/TileTile!gradients_1/pi/Sum_2_grad/Reshape"gradients_1/pi/Sum_2_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:         
g
gradients_1/pi/sub_4_grad/ShapeShapepi/add_9*
out_type0*
T0*
_output_shapes
:
p
!gradients_1/pi/sub_4_grad/Shape_1Shapepi/log_std/read*
T0*
out_type0*
_output_shapes
:
╔
/gradients_1/pi/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/sub_4_grad/Shape!gradients_1/pi/sub_4_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╡
gradients_1/pi/sub_4_grad/SumSumgradients_1/pi/Sum_2_grad/Tile/gradients_1/pi/sub_4_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
м
!gradients_1/pi/sub_4_grad/ReshapeReshapegradients_1/pi/sub_4_grad/Sumgradients_1/pi/sub_4_grad/Shape*'
_output_shapes
:         *
Tshape0*
T0
v
gradients_1/pi/sub_4_grad/NegNeggradients_1/pi/Sum_2_grad/Tile*
T0*'
_output_shapes
:         
╕
gradients_1/pi/sub_4_grad/Sum_1Sumgradients_1/pi/sub_4_grad/Neg1gradients_1/pi/sub_4_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:*

Tidx0
е
#gradients_1/pi/sub_4_grad/Reshape_1Reshapegradients_1/pi/sub_4_grad/Sum_1!gradients_1/pi/sub_4_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
g
gradients_1/pi/add_9_grad/ShapeShapepi/mul_7*
_output_shapes
:*
out_type0*
T0
q
!gradients_1/pi/add_9_grad/Shape_1Shapepi/Placeholder_1*
T0*
_output_shapes
:*
out_type0
╔
/gradients_1/pi/add_9_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/add_9_grad/Shape!gradients_1/pi/add_9_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╕
gradients_1/pi/add_9_grad/SumSum!gradients_1/pi/sub_4_grad/Reshape/gradients_1/pi/add_9_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
м
!gradients_1/pi/add_9_grad/ReshapeReshapegradients_1/pi/add_9_grad/Sumgradients_1/pi/add_9_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
╝
gradients_1/pi/add_9_grad/Sum_1Sum!gradients_1/pi/sub_4_grad/Reshape1gradients_1/pi/add_9_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
▓
#gradients_1/pi/add_9_grad/Reshape_1Reshapegradients_1/pi/add_9_grad/Sum_1!gradients_1/pi/add_9_grad/Shape_1*
Tshape0*'
_output_shapes
:         *
T0
g
gradients_1/pi/mul_7_grad/ShapeShape
pi/mul_7/x*
T0*
_output_shapes
: *
out_type0
i
!gradients_1/pi/mul_7_grad/Shape_1Shapepi/sub_3*
T0*
out_type0*
_output_shapes
:
╔
/gradients_1/pi/mul_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/mul_7_grad/Shape!gradients_1/pi/mul_7_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Г
gradients_1/pi/mul_7_grad/MulMul!gradients_1/pi/add_9_grad/Reshapepi/sub_3*
T0*'
_output_shapes
:         
┤
gradients_1/pi/mul_7_grad/SumSumgradients_1/pi/mul_7_grad/Mul/gradients_1/pi/mul_7_grad/BroadcastGradientArgs*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
Ы
!gradients_1/pi/mul_7_grad/ReshapeReshapegradients_1/pi/mul_7_grad/Sumgradients_1/pi/mul_7_grad/Shape*
_output_shapes
: *
Tshape0*
T0
З
gradients_1/pi/mul_7_grad/Mul_1Mul
pi/mul_7/x!gradients_1/pi/add_9_grad/Reshape*
T0*'
_output_shapes
:         
║
gradients_1/pi/mul_7_grad/Sum_1Sumgradients_1/pi/mul_7_grad/Mul_11gradients_1/pi/mul_7_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
▓
#gradients_1/pi/mul_7_grad/Reshape_1Reshapegradients_1/pi/mul_7_grad/Sum_1!gradients_1/pi/mul_7_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
k
gradients_1/pi/sub_3_grad/ShapeShapepi/truediv_2*
T0*
out_type0*
_output_shapes
:
i
!gradients_1/pi/sub_3_grad/Shape_1Shape
pi/sub_3/y*
_output_shapes
: *
T0*
out_type0
╔
/gradients_1/pi/sub_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/sub_3_grad/Shape!gradients_1/pi/sub_3_grad/Shape_1*
T0*2
_output_shapes 
:         :         
║
gradients_1/pi/sub_3_grad/SumSum#gradients_1/pi/mul_7_grad/Reshape_1/gradients_1/pi/sub_3_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
м
!gradients_1/pi/sub_3_grad/ReshapeReshapegradients_1/pi/sub_3_grad/Sumgradients_1/pi/sub_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
{
gradients_1/pi/sub_3_grad/NegNeg#gradients_1/pi/mul_7_grad/Reshape_1*'
_output_shapes
:         *
T0
╕
gradients_1/pi/sub_3_grad/Sum_1Sumgradients_1/pi/sub_3_grad/Neg1gradients_1/pi/sub_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
б
#gradients_1/pi/sub_3_grad/Reshape_1Reshapegradients_1/pi/sub_3_grad/Sum_1!gradients_1/pi/sub_3_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
k
#gradients_1/pi/truediv_2_grad/ShapeShapepi/add_7*
T0*
_output_shapes
:*
out_type0
m
%gradients_1/pi/truediv_2_grad/Shape_1Shapepi/add_8*
_output_shapes
:*
T0*
out_type0
╒
3gradients_1/pi/truediv_2_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients_1/pi/truediv_2_grad/Shape%gradients_1/pi/truediv_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
П
%gradients_1/pi/truediv_2_grad/RealDivRealDiv!gradients_1/pi/sub_3_grad/Reshapepi/add_8*'
_output_shapes
:         *
T0
─
!gradients_1/pi/truediv_2_grad/SumSum%gradients_1/pi/truediv_2_grad/RealDiv3gradients_1/pi/truediv_2_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
╕
%gradients_1/pi/truediv_2_grad/ReshapeReshape!gradients_1/pi/truediv_2_grad/Sum#gradients_1/pi/truediv_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
d
!gradients_1/pi/truediv_2_grad/NegNegpi/add_7*'
_output_shapes
:         *
T0
С
'gradients_1/pi/truediv_2_grad/RealDiv_1RealDiv!gradients_1/pi/truediv_2_grad/Negpi/add_8*'
_output_shapes
:         *
T0
Ч
'gradients_1/pi/truediv_2_grad/RealDiv_2RealDiv'gradients_1/pi/truediv_2_grad/RealDiv_1pi/add_8*'
_output_shapes
:         *
T0
ж
!gradients_1/pi/truediv_2_grad/mulMul!gradients_1/pi/sub_3_grad/Reshape'gradients_1/pi/truediv_2_grad/RealDiv_2*
T0*'
_output_shapes
:         
─
#gradients_1/pi/truediv_2_grad/Sum_1Sum!gradients_1/pi/truediv_2_grad/mul5gradients_1/pi/truediv_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
╛
'gradients_1/pi/truediv_2_grad/Reshape_1Reshape#gradients_1/pi/truediv_2_grad/Sum_1%gradients_1/pi/truediv_2_grad/Shape_1*'
_output_shapes
:         *
Tshape0*
T0
g
gradients_1/pi/add_7_grad/ShapeShapepi/pow_2*
out_type0*
_output_shapes
:*
T0
i
!gradients_1/pi/add_7_grad/Shape_1Shapepi/Exp_3*
out_type0*
T0*
_output_shapes
:
╔
/gradients_1/pi/add_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/add_7_grad/Shape!gradients_1/pi/add_7_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╝
gradients_1/pi/add_7_grad/SumSum%gradients_1/pi/truediv_2_grad/Reshape/gradients_1/pi/add_7_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
м
!gradients_1/pi/add_7_grad/ReshapeReshapegradients_1/pi/add_7_grad/Sumgradients_1/pi/add_7_grad/Shape*'
_output_shapes
:         *
Tshape0*
T0
└
gradients_1/pi/add_7_grad/Sum_1Sum%gradients_1/pi/truediv_2_grad/Reshape1gradients_1/pi/add_7_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:*

Tidx0
е
#gradients_1/pi/add_7_grad/Reshape_1Reshapegradients_1/pi/add_7_grad/Sum_1!gradients_1/pi/add_7_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
g
gradients_1/pi/pow_2_grad/ShapeShapepi/sub_2*
_output_shapes
:*
out_type0*
T0
i
!gradients_1/pi/pow_2_grad/Shape_1Shape
pi/pow_2/y*
_output_shapes
: *
out_type0*
T0
╔
/gradients_1/pi/pow_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/pow_2_grad/Shape!gradients_1/pi/pow_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Е
gradients_1/pi/pow_2_grad/mulMul!gradients_1/pi/add_7_grad/Reshape
pi/pow_2/y*
T0*'
_output_shapes
:         
d
gradients_1/pi/pow_2_grad/sub/yConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
r
gradients_1/pi/pow_2_grad/subSub
pi/pow_2/ygradients_1/pi/pow_2_grad/sub/y*
T0*
_output_shapes
: 

gradients_1/pi/pow_2_grad/PowPowpi/sub_2gradients_1/pi/pow_2_grad/sub*'
_output_shapes
:         *
T0
Ц
gradients_1/pi/pow_2_grad/mul_1Mulgradients_1/pi/pow_2_grad/mulgradients_1/pi/pow_2_grad/Pow*'
_output_shapes
:         *
T0
╢
gradients_1/pi/pow_2_grad/SumSumgradients_1/pi/pow_2_grad/mul_1/gradients_1/pi/pow_2_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
м
!gradients_1/pi/pow_2_grad/ReshapeReshapegradients_1/pi/pow_2_grad/Sumgradients_1/pi/pow_2_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
h
#gradients_1/pi/pow_2_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Н
!gradients_1/pi/pow_2_grad/GreaterGreaterpi/sub_2#gradients_1/pi/pow_2_grad/Greater/y*'
_output_shapes
:         *
T0
q
)gradients_1/pi/pow_2_grad/ones_like/ShapeShapepi/sub_2*
out_type0*
_output_shapes
:*
T0
n
)gradients_1/pi/pow_2_grad/ones_like/ConstConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
┼
#gradients_1/pi/pow_2_grad/ones_likeFill)gradients_1/pi/pow_2_grad/ones_like/Shape)gradients_1/pi/pow_2_grad/ones_like/Const*'
_output_shapes
:         *

index_type0*
T0
о
 gradients_1/pi/pow_2_grad/SelectSelect!gradients_1/pi/pow_2_grad/Greaterpi/sub_2#gradients_1/pi/pow_2_grad/ones_like*'
_output_shapes
:         *
T0
x
gradients_1/pi/pow_2_grad/LogLog gradients_1/pi/pow_2_grad/Select*
T0*'
_output_shapes
:         
m
$gradients_1/pi/pow_2_grad/zeros_like	ZerosLikepi/sub_2*'
_output_shapes
:         *
T0
╞
"gradients_1/pi/pow_2_grad/Select_1Select!gradients_1/pi/pow_2_grad/Greatergradients_1/pi/pow_2_grad/Log$gradients_1/pi/pow_2_grad/zeros_like*
T0*'
_output_shapes
:         
Е
gradients_1/pi/pow_2_grad/mul_2Mul!gradients_1/pi/add_7_grad/Reshapepi/pow_2*
T0*'
_output_shapes
:         
Э
gradients_1/pi/pow_2_grad/mul_3Mulgradients_1/pi/pow_2_grad/mul_2"gradients_1/pi/pow_2_grad/Select_1*'
_output_shapes
:         *
T0
║
gradients_1/pi/pow_2_grad/Sum_1Sumgradients_1/pi/pow_2_grad/mul_31gradients_1/pi/pow_2_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
б
#gradients_1/pi/pow_2_grad/Reshape_1Reshapegradients_1/pi/pow_2_grad/Sum_1!gradients_1/pi/pow_2_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
x
gradients_1/pi/Exp_3_grad/mulMul#gradients_1/pi/add_7_grad/Reshape_1pi/Exp_3*
T0*
_output_shapes
:
m
gradients_1/pi/sub_2_grad/ShapeShapepi/Placeholder*
out_type0*
_output_shapes
:*
T0
s
!gradients_1/pi/sub_2_grad/Shape_1Shapepi/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
╔
/gradients_1/pi/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/sub_2_grad/Shape!gradients_1/pi/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╕
gradients_1/pi/sub_2_grad/SumSum!gradients_1/pi/pow_2_grad/Reshape/gradients_1/pi/sub_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
!gradients_1/pi/sub_2_grad/ReshapeReshapegradients_1/pi/sub_2_grad/Sumgradients_1/pi/sub_2_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
y
gradients_1/pi/sub_2_grad/NegNeg!gradients_1/pi/pow_2_grad/Reshape*
T0*'
_output_shapes
:         
╕
gradients_1/pi/sub_2_grad/Sum_1Sumgradients_1/pi/sub_2_grad/Neg1gradients_1/pi/sub_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
▓
#gradients_1/pi/sub_2_grad/Reshape_1Reshapegradients_1/pi/sub_2_grad/Sum_1!gradients_1/pi/sub_2_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
y
gradients_1/pi/mul_5_grad/MulMulgradients_1/pi/Exp_3_grad/mulpi/log_std/read*
T0*
_output_shapes
:
y
/gradients_1/pi/mul_5_grad/Sum/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0
▓
gradients_1/pi/mul_5_grad/SumSumgradients_1/pi/mul_5_grad/Mul/gradients_1/pi/mul_5_grad/Sum/reduction_indices*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
j
'gradients_1/pi/mul_5_grad/Reshape/shapeConst*
_output_shapes
: *
valueB *
dtype0
г
!gradients_1/pi/mul_5_grad/ReshapeReshapegradients_1/pi/mul_5_grad/Sum'gradients_1/pi/mul_5_grad/Reshape/shape*
_output_shapes
: *
Tshape0*
T0
v
gradients_1/pi/mul_5_grad/Mul_1Mul
pi/mul_5/xgradients_1/pi/Exp_3_grad/mul*
T0*
_output_shapes
:
Я
/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_1/pi/sub_2_grad/Reshape_1*
T0*
data_formatNHWC*
_output_shapes
:
─
gradients_1/AddNAddN#gradients_1/pi/sub_4_grad/Reshape_1gradients_1/pi/mul_5_grad/Mul_1*
T0*6
_class,
*(loc:@gradients_1/pi/sub_4_grad/Reshape_1*
_output_shapes
:*
N
╔
)gradients_1/pi/dense_2/MatMul_grad/MatMulMatMul#gradients_1/pi/sub_2_grad/Reshape_1pi/dense_2/kernel/read*
transpose_b(*(
_output_shapes
:         А*
transpose_a( *
T0
╗
+gradients_1/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh#gradients_1/pi/sub_2_grad/Reshape_1*
T0*
_output_shapes
:	А*
transpose_a(*
transpose_b( 
д
)gradients_1/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh)gradients_1/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:         А*
T0
ж
/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_1/pi/dense_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
╧
)gradients_1/pi/dense_1/MatMul_grad/MatMulMatMul)gradients_1/pi/dense_1/Tanh_grad/TanhGradpi/dense_1/kernel/read*
transpose_a( *(
_output_shapes
:         А*
T0*
transpose_b(
└
+gradients_1/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh)gradients_1/pi/dense_1/Tanh_grad/TanhGrad*
transpose_a(*
T0*
transpose_b( * 
_output_shapes
:
АА
а
'gradients_1/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh)gradients_1/pi/dense_1/MatMul_grad/MatMul*(
_output_shapes
:         А*
T0
в
-gradients_1/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_1/pi/dense/Tanh_grad/TanhGrad*
_output_shapes	
:А*
T0*
data_formatNHWC
╚
'gradients_1/pi/dense/MatMul_grad/MatMulMatMul'gradients_1/pi/dense/Tanh_grad/TanhGradpi/dense/kernel/read*
transpose_a( *'
_output_shapes
:         <*
transpose_b(*
T0
╣
)gradients_1/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder'gradients_1/pi/dense/Tanh_grad/TanhGrad*
T0*
transpose_a(*
transpose_b( *
_output_shapes
:	<А
b
Reshape_7/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
Д
	Reshape_7Reshape)gradients_1/pi/dense/MatMul_grad/MatMul_1Reshape_7/shape*
Tshape0*
T0*
_output_shapes	
:Аx
b
Reshape_8/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
И
	Reshape_8Reshape-gradients_1/pi/dense/BiasAdd_grad/BiasAddGradReshape_8/shape*
_output_shapes	
:А*
T0*
Tshape0
b
Reshape_9/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
З
	Reshape_9Reshape+gradients_1/pi/dense_1/MatMul_grad/MatMul_1Reshape_9/shape*
Tshape0*
_output_shapes

:АА*
T0
c
Reshape_10/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
М

Reshape_10Reshape/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGradReshape_10/shape*
Tshape0*
_output_shapes	
:А*
T0
c
Reshape_11/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
И

Reshape_11Reshape+gradients_1/pi/dense_2/MatMul_grad/MatMul_1Reshape_11/shape*
T0*
Tshape0*
_output_shapes	
:А
c
Reshape_12/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
Л

Reshape_12Reshape/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGradReshape_12/shape*
_output_shapes
:*
Tshape0*
T0
c
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
l

Reshape_13Reshapegradients_1/AddNReshape_13/shape*
T0*
Tshape0*
_output_shapes
:
O
concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
░
concat_1ConcatV2	Reshape_7	Reshape_8	Reshape_9
Reshape_10
Reshape_11
Reshape_12
Reshape_13concat_1/axis*

Tidx0*
T0*
_output_shapes

:ДА*
N
Z
Placeholder_9Placeholder*
_output_shapes

:ДА*
dtype0*
shape:ДА
L
mul_3Mulconcat_1Placeholder_9*
_output_shapes

:ДА*
T0
Q
Const_3Const*
dtype0*
valueB: *
_output_shapes
:
X
SumSummul_3Const_3*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
T
gradients_2/ShapeConst*
valueB *
_output_shapes
: *
dtype0
Z
gradients_2/grad_ys_0Const*
dtype0*
valueB
 *  А?*
_output_shapes
: 
u
gradients_2/FillFillgradients_2/Shapegradients_2/grad_ys_0*
T0*
_output_shapes
: *

index_type0
l
"gradients_2/Sum_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Р
gradients_2/Sum_grad/ReshapeReshapegradients_2/Fill"gradients_2/Sum_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
f
gradients_2/Sum_grad/ConstConst*
valueB:ДА*
dtype0*
_output_shapes
:
Ф
gradients_2/Sum_grad/TileTilegradients_2/Sum_grad/Reshapegradients_2/Sum_grad/Const*

Tmultiples0*
_output_shapes

:ДА*
T0
r
gradients_2/mul_3_grad/MulMulgradients_2/Sum_grad/TilePlaceholder_9*
_output_shapes

:ДА*
T0
o
gradients_2/mul_3_grad/Mul_1Mulgradients_2/Sum_grad/Tileconcat_1*
_output_shapes

:ДА*
T0
`
gradients_2/concat_1_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
y
gradients_2/concat_1_grad/modFloorModconcat_1/axisgradients_2/concat_1_grad/Rank*
T0*
_output_shapes
: 
j
gradients_2/concat_1_grad/ShapeConst*
dtype0*
valueB:Аx*
_output_shapes
:
l
!gradients_2/concat_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:А
m
!gradients_2/concat_1_grad/Shape_2Const*
_output_shapes
:*
valueB:АА*
dtype0
l
!gradients_2/concat_1_grad/Shape_3Const*
dtype0*
valueB:А*
_output_shapes
:
l
!gradients_2/concat_1_grad/Shape_4Const*
valueB:А*
dtype0*
_output_shapes
:
k
!gradients_2/concat_1_grad/Shape_5Const*
valueB:*
dtype0*
_output_shapes
:
k
!gradients_2/concat_1_grad/Shape_6Const*
dtype0*
_output_shapes
:*
valueB:
С
&gradients_2/concat_1_grad/ConcatOffsetConcatOffsetgradients_2/concat_1_grad/modgradients_2/concat_1_grad/Shape!gradients_2/concat_1_grad/Shape_1!gradients_2/concat_1_grad/Shape_2!gradients_2/concat_1_grad/Shape_3!gradients_2/concat_1_grad/Shape_4!gradients_2/concat_1_grad/Shape_5!gradients_2/concat_1_grad/Shape_6*
N*>
_output_shapes,
*:::::::
└
gradients_2/concat_1_grad/SliceSlicegradients_2/mul_3_grad/Mul&gradients_2/concat_1_grad/ConcatOffsetgradients_2/concat_1_grad/Shape*
Index0*
T0*
_output_shapes	
:Аx
╞
!gradients_2/concat_1_grad/Slice_1Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:1!gradients_2/concat_1_grad/Shape_1*
T0*
_output_shapes	
:А*
Index0
╟
!gradients_2/concat_1_grad/Slice_2Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:2!gradients_2/concat_1_grad/Shape_2*
T0*
_output_shapes

:АА*
Index0
╞
!gradients_2/concat_1_grad/Slice_3Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:3!gradients_2/concat_1_grad/Shape_3*
_output_shapes	
:А*
T0*
Index0
╞
!gradients_2/concat_1_grad/Slice_4Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:4!gradients_2/concat_1_grad/Shape_4*
_output_shapes	
:А*
Index0*
T0
┼
!gradients_2/concat_1_grad/Slice_5Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:5!gradients_2/concat_1_grad/Shape_5*
_output_shapes
:*
Index0*
T0
┼
!gradients_2/concat_1_grad/Slice_6Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:6!gradients_2/concat_1_grad/Shape_6*
Index0*
T0*
_output_shapes
:
q
 gradients_2/Reshape_7_grad/ShapeConst*
dtype0*
valueB"<      *
_output_shapes
:
и
"gradients_2/Reshape_7_grad/ReshapeReshapegradients_2/concat_1_grad/Slice gradients_2/Reshape_7_grad/Shape*
_output_shapes
:	<А*
Tshape0*
T0
k
 gradients_2/Reshape_8_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:А
ж
"gradients_2/Reshape_8_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_1 gradients_2/Reshape_8_grad/Shape*
T0*
Tshape0*
_output_shapes	
:А
q
 gradients_2/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
л
"gradients_2/Reshape_9_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_2 gradients_2/Reshape_9_grad/Shape* 
_output_shapes
:
АА*
Tshape0*
T0
l
!gradients_2/Reshape_10_grad/ShapeConst*
valueB:А*
_output_shapes
:*
dtype0
и
#gradients_2/Reshape_10_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_3!gradients_2/Reshape_10_grad/Shape*
Tshape0*
_output_shapes	
:А*
T0
r
!gradients_2/Reshape_11_grad/ShapeConst*
_output_shapes
:*
valueB"      *
dtype0
м
#gradients_2/Reshape_11_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_4!gradients_2/Reshape_11_grad/Shape*
T0*
Tshape0*
_output_shapes
:	А
k
!gradients_2/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
з
#gradients_2/Reshape_12_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_5!gradients_2/Reshape_12_grad/Shape*
_output_shapes
:*
Tshape0*
T0
k
!gradients_2/Reshape_13_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:
з
#gradients_2/Reshape_13_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_6!gradients_2/Reshape_13_grad/Shape*
_output_shapes
:*
Tshape0*
T0
Ё
Agradients_2/gradients_1/pi/dense/MatMul_grad/MatMul_1_grad/MatMulMatMul'gradients_1/pi/dense/Tanh_grad/TanhGrad"gradients_2/Reshape_7_grad/Reshape*'
_output_shapes
:         <*
transpose_a( *
T0*
transpose_b(
╫
Cgradients_2/gradients_1/pi/dense/MatMul_grad/MatMul_1_grad/MatMul_1MatMulPlaceholder"gradients_2/Reshape_7_grad/Reshape*(
_output_shapes
:         А*
T0*
transpose_b( *
transpose_a( 
л
Dgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ShapeShape'gradients_1/pi/dense/Tanh_grad/TanhGrad*
out_type0*
_output_shapes
:*
T0
С
Fgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/Shape_1Const*
valueB:А*
dtype0*
_output_shapes
:
Ь
Rgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
з
Tgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
         
Ю
Tgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
╕
Lgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_sliceStridedSliceDgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ShapeRgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackTgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Tgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2*
T0*
new_axis_mask *
end_mask *

begin_mask*
Index0*
_output_shapes
:*
ellipsis_mask *
shrink_axis_mask 
Ш
Ngradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapeConst*
_output_shapes
:*
valueB:*
dtype0
Р
Ngradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_like/ConstConst*
_output_shapes
: *
value	B :*
dtype0
з
Hgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_likeFillNgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapeNgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_like/Const*
T0*

index_type0*
_output_shapes
:
М
Jgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
щ
Egradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concatConcatV2Hgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_likeFgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/Shape_1Jgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat/axis*

Tidx0*
N*
_output_shapes
:*
T0
Ю
Tgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
й
Vgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:
         *
dtype0
а
Vgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
└
Ngradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1StridedSliceDgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ShapeTgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackVgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Vgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2*
shrink_axis_mask *
Index0*

begin_mask*
end_mask *
T0*
ellipsis_mask *
_output_shapes
:*
new_axis_mask 
Ъ
Pgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Const*
_output_shapes
:*
valueB:*
dtype0
О
Lgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
¤
Ggradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1ConcatV2Ngradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1Pgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Lgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1/axis*

Tidx0*
N*
_output_shapes
:*
T0
Ї
Fgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ReshapeReshape"gradients_2/Reshape_8_grad/ReshapeEgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat*
_output_shapes
:	А*
T0*
Tshape0
б
Cgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/TileTileFgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ReshapeGgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1*

Tmultiples0*
T0*(
_output_shapes
:         А
ї
Cgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMulMatMul)gradients_1/pi/dense_1/Tanh_grad/TanhGrad"gradients_2/Reshape_9_grad/Reshape*
transpose_b(*(
_output_shapes
:         А*
transpose_a( *
T0
█
Egradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMul_1MatMulpi/dense/Tanh"gradients_2/Reshape_9_grad/Reshape*(
_output_shapes
:         А*
T0*
transpose_b( *
transpose_a( 
п
Fgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ShapeShape)gradients_1/pi/dense_1/Tanh_grad/TanhGrad*
_output_shapes
:*
T0*
out_type0
У
Hgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/Shape_1Const*
dtype0*
valueB:А*
_output_shapes
:
Ю
Tgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
й
Vgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Const*
dtype0*
valueB:
         *
_output_shapes
:
а
Vgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┬
Ngradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_sliceStridedSliceFgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ShapeTgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackVgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Vgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2*
shrink_axis_mask *
Index0*
_output_shapes
:*
T0*
end_mask *
ellipsis_mask *

begin_mask*
new_axis_mask 
Ъ
Pgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
Т
Pgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_like/ConstConst*
_output_shapes
: *
value	B :*
dtype0
н
Jgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_likeFillPgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapePgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_like/Const*
_output_shapes
:*

index_type0*
T0
О
Lgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
ё
Ggradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concatConcatV2Jgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_likeHgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/Shape_1Lgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat/axis*
T0*

Tidx0*
N*
_output_shapes
:
а
Vgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0
л
Xgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
в
Xgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╩
Pgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1StridedSliceFgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ShapeVgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackXgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Xgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2*
ellipsis_mask *
end_mask *
Index0*

begin_mask*
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
T0
Ь
Rgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Const*
valueB:*
dtype0*
_output_shapes
:
Р
Ngradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
Е
Igradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1ConcatV2Pgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1Rgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Ngradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
∙
Hgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ReshapeReshape#gradients_2/Reshape_10_grad/ReshapeGgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat*
T0*
Tshape0*
_output_shapes
:	А
з
Egradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/TileTileHgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ReshapeIgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1*
T0*

Tmultiples0*(
_output_shapes
:         А
Ё
Cgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMulMatMul#gradients_1/pi/sub_2_grad/Reshape_1#gradients_2/Reshape_11_grad/Reshape*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:         А
▌
Egradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMul_1MatMulpi/dense_1/Tanh#gradients_2/Reshape_11_grad/Reshape*'
_output_shapes
:         *
T0*
transpose_a( *
transpose_b( 
й
Fgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ShapeShape#gradients_1/pi/sub_2_grad/Reshape_1*
out_type0*
_output_shapes
:*
T0
Т
Hgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
Ю
Tgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
й
Vgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Const*
valueB:
         *
_output_shapes
:*
dtype0
а
Vgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
┬
Ngradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_sliceStridedSliceFgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ShapeTgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackVgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Vgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2*
_output_shapes
:*
end_mask *

begin_mask*
shrink_axis_mask *
Index0*
T0*
ellipsis_mask *
new_axis_mask 
Ъ
Pgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapeConst*
valueB:*
_output_shapes
:*
dtype0
Т
Pgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_like/ConstConst*
value	B :*
_output_shapes
: *
dtype0
н
Jgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_likeFillPgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapePgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_like/Const*
T0*
_output_shapes
:*

index_type0
О
Lgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
ё
Ggradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concatConcatV2Jgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_likeHgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/Shape_1Lgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat/axis*
T0*

Tidx0*
_output_shapes
:*
N
а
Vgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
л
Xgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Const*
dtype0*
valueB:
         *
_output_shapes
:
в
Xgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╩
Pgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1StridedSliceFgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ShapeVgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackXgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Xgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2*
shrink_axis_mask *
end_mask *
Index0*

begin_mask*
new_axis_mask *
T0*
_output_shapes
:*
ellipsis_mask 
Ь
Rgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:
Р
Ngradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
Е
Igradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1ConcatV2Pgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1Rgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Ngradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1/axis*
T0*
N*

Tidx0*
_output_shapes
:
°
Hgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ReshapeReshape#gradients_2/Reshape_12_grad/ReshapeGgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat*
T0*
_output_shapes

:*
Tshape0
ж
Egradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/TileTileHgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ReshapeIgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1*'
_output_shapes
:         *

Tmultiples0*
T0
╢
gradients_2/AddNAddNCgradients_2/gradients_1/pi/dense/MatMul_grad/MatMul_1_grad/MatMul_1Cgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/Tile*V
_classL
JHloc:@gradients_2/gradients_1/pi/dense/MatMul_grad/MatMul_1_grad/MatMul_1*
T0*(
_output_shapes
:         А*
N
Ц
>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul/yConst^gradients_2/AddN*
dtype0*
_output_shapes
: *
valueB
 *   └
╚
<gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mulMulgradients_2/AddN>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul/y*(
_output_shapes
:         А*
T0
с
>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul_1Mul<gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul)gradients_1/pi/dense_1/MatMul_grad/MatMul*(
_output_shapes
:         А*
T0
╟
>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul_2Mul>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul_1pi/dense/Tanh*(
_output_shapes
:         А*
T0
б
Agradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/TanhGradTanhGradpi/dense/Tanhgradients_2/AddN*
T0*(
_output_shapes
:         А
д
4gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/MulMul#gradients_2/Reshape_13_grad/Reshapegradients_1/pi/Exp_3_grad/mul*
_output_shapes
:*
T0
Р
Fgradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Sum/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
ў
4gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/SumSum4gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/MulFgradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Б
>gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Reshape/shapeConst*
dtype0*
valueB *
_output_shapes
: 
ш
8gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/ReshapeReshape4gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Sum>gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
: 
У
6gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Mul_1Mul
pi/mul_5/x#gradients_2/Reshape_13_grad/Reshape*
T0*
_output_shapes
:
 
Agradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_grad/MatMulMatMulAgradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/TanhGradpi/dense_1/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
М
Cgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_grad/MatMul_1MatMulAgradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/TanhGrad)gradients_1/pi/dense_1/Tanh_grad/TanhGrad* 
_output_shapes
:
АА*
transpose_b( *
T0*
transpose_a(
а
2gradients_2/gradients_1/pi/Exp_3_grad/mul_grad/MulMul6gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Mul_1pi/Exp_3*
T0*
_output_shapes
:
╜
4gradients_2/gradients_1/pi/Exp_3_grad/mul_grad/Mul_1Mul6gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Mul_1#gradients_1/pi/add_7_grad/Reshape_1*
_output_shapes
:*
T0
Б
gradients_2/AddN_1AddNEgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMul_1Egradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/TileAgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_grad/MatMul*
T0*X
_classN
LJloc:@gradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMul_1*(
_output_shapes
:         А*
N
Ъ
@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul/yConst^gradients_2/AddN_1*
dtype0*
valueB
 *   └*
_output_shapes
: 
╬
>gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mulMulgradients_2/AddN_1@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul/y*(
_output_shapes
:         А*
T0
х
@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul_1Mul>gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul)gradients_1/pi/dense_2/MatMul_grad/MatMul*
T0*(
_output_shapes
:         А
═
@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul_2Mul@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul_1pi/dense_1/Tanh*(
_output_shapes
:         А*
T0
з
Cgradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/TanhGradTanhGradpi/dense_1/Tanhgradients_2/AddN_1*
T0*(
_output_shapes
:         А
Й
gradients_2/pi/Exp_3_grad/mulMul4gradients_2/gradients_1/pi/Exp_3_grad/mul_grad/Mul_1pi/Exp_3*
T0*
_output_shapes
:
А
Agradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_grad/MatMulMatMulCgradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/TanhGradpi/dense_2/kernel/read*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:         
З
Cgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_grad/MatMul_1MatMulCgradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/TanhGrad#gradients_1/pi/sub_2_grad/Reshape_1*
_output_shapes
:	А*
T0*
transpose_a(*
transpose_b( 
y
gradients_2/pi/mul_5_grad/MulMulgradients_2/pi/Exp_3_grad/mulpi/log_std/read*
T0*
_output_shapes
:
y
/gradients_2/pi/mul_5_grad/Sum/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0
▓
gradients_2/pi/mul_5_grad/SumSumgradients_2/pi/mul_5_grad/Mul/gradients_2/pi/mul_5_grad/Sum/reduction_indices*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
j
'gradients_2/pi/mul_5_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
г
!gradients_2/pi/mul_5_grad/ReshapeReshapegradients_2/pi/mul_5_grad/Sum'gradients_2/pi/mul_5_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
: 
v
gradients_2/pi/mul_5_grad/Mul_1Mul
pi/mul_5/xgradients_2/pi/Exp_3_grad/mul*
_output_shapes
:*
T0
А
gradients_2/AddN_2AddNEgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMul_1Egradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/TileAgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_grad/MatMul*'
_output_shapes
:         *X
_classN
LJloc:@gradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMul_1*
T0*
N
в
:gradients_2/gradients_1/pi/sub_2_grad/Reshape_1_grad/ShapeShapegradients_1/pi/sub_2_grad/Sum_1*
out_type0*
T0*#
_output_shapes
:         
╚
<gradients_2/gradients_1/pi/sub_2_grad/Reshape_1_grad/ReshapeReshapegradients_2/AddN_2:gradients_2/gradients_1/pi/sub_2_grad/Reshape_1_grad/Shape*
Tshape0*
T0*
_output_shapes
:
У
6gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/ShapeShapegradients_1/pi/sub_2_grad/Neg*
T0*
_output_shapes
:*
out_type0
┬
5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/SizeConst*
value	B :*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
_output_shapes
: *
dtype0
а
4gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/addAddV21gradients_1/pi/sub_2_grad/BroadcastGradientArgs:15gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Size*#
_output_shapes
:         *
T0*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape
ж
4gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/modFloorMod4gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/add5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Size*
T0*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*#
_output_shapes
:         
ў
8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape_1Shape4gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/mod*
out_type0*
T0*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
_output_shapes
:
╔
<gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/range/startConst*
value	B : *I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
╔
<gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape
х
6gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/rangeRange<gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/range/start5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Size<gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/range/delta*

Tidx0*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
_output_shapes
:
╚
;gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Fill/valueConst*
_output_shapes
: *
value	B :*
dtype0*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape
┐
5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/FillFill8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape_1;gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Fill/value*#
_output_shapes
:         *I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*

index_type0*
T0
о
>gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/DynamicStitchDynamicStitch6gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/range4gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/mod6gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Fill*
T0*#
_output_shapes
:         *
N*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape
╟
:gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Maximum/yConst*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
_output_shapes
: *
value	B :*
dtype0
╕
8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/MaximumMaximum>gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/DynamicStitch:gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Maximum/y*#
_output_shapes
:         *
T0*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape
з
9gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/floordivFloorDiv6gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Maximum*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
T0*
_output_shapes
:
Є
8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/ReshapeReshape<gradients_2/gradients_1/pi/sub_2_grad/Reshape_1_grad/Reshape>gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Ў
5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/TileTile8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Reshape9gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/floordiv*

Tmultiples0*'
_output_shapes
:         *
T0
в
2gradients_2/gradients_1/pi/sub_2_grad/Neg_grad/NegNeg5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Tile*
T0*'
_output_shapes
:         
Ю
8gradients_2/gradients_1/pi/pow_2_grad/Reshape_grad/ShapeShapegradients_1/pi/pow_2_grad/Sum*
T0*
out_type0*#
_output_shapes
:         
ф
:gradients_2/gradients_1/pi/pow_2_grad/Reshape_grad/ReshapeReshape2gradients_2/gradients_1/pi/sub_2_grad/Neg_grad/Neg8gradients_2/gradients_1/pi/pow_2_grad/Reshape_grad/Shape*
_output_shapes
:*
Tshape0*
T0
У
4gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/ShapeShapegradients_1/pi/pow_2_grad/mul_1*
T0*
out_type0*
_output_shapes
:
╛
3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/SizeConst*
value	B :*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Ш
2gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/addAddV2/gradients_1/pi/pow_2_grad/BroadcastGradientArgs3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Size*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
T0*#
_output_shapes
:         
Ю
2gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/modFloorMod2gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/add3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Size*
T0*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*#
_output_shapes
:         
ё
6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape_1Shape2gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/mod*
T0*
out_type0*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
_output_shapes
:
┼
:gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/range/startConst*
dtype0*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
_output_shapes
: *
value	B : 
┼
:gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/range/deltaConst*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
value	B :*
_output_shapes
: *
dtype0
█
4gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/rangeRange:gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/range/start3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Size:gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/range/delta*
_output_shapes
:*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*

Tidx0
─
9gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Fill/valueConst*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
╖
3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/FillFill6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape_19gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Fill/value*
T0*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*#
_output_shapes
:         *

index_type0
в
<gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/DynamicStitchDynamicStitch4gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/range2gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/mod4gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Fill*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*#
_output_shapes
:         *
T0*
N
├
8gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Maximum/yConst*
dtype0*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
_output_shapes
: *
value	B :
░
6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/MaximumMaximum<gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/DynamicStitch8gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Maximum/y*
T0*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*#
_output_shapes
:         
Я
7gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/floordivFloorDiv4gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Maximum*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
T0*
_output_shapes
:
ь
6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/ReshapeReshape:gradients_2/gradients_1/pi/pow_2_grad/Reshape_grad/Reshape<gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Ё
3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/TileTile6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Reshape7gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:         
У
6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/ShapeShapegradients_1/pi/pow_2_grad/mul*
_output_shapes
:*
out_type0*
T0
Х
8gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Shape_1Shapegradients_1/pi/pow_2_grad/Pow*
_output_shapes
:*
out_type0*
T0
О
Fgradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Shape8gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┴
4gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/MulMul3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Tilegradients_1/pi/pow_2_grad/Pow*'
_output_shapes
:         *
T0
∙
4gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/SumSum4gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/MulFgradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
ё
8gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/ReshapeReshape4gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Sum6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
├
6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Mul_1Mulgradients_1/pi/pow_2_grad/mul3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Tile*
T0*'
_output_shapes
:         
 
6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Sum_1Sum6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Mul_1Hgradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
ў
:gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Reshape_1Reshape6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Sum_18gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
|
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ShapeShapepi/sub_2*
_output_shapes
:*
T0*
out_type0
С
6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Shape_1Shapegradients_1/pi/pow_2_grad/sub*
T0*
_output_shapes
: *
out_type0
И
Dgradients_2/gradients_1/pi/pow_2_grad/Pow_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Shape6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╞
2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mulMul:gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Reshape_1gradients_1/pi/pow_2_grad/sub*
T0*'
_output_shapes
:         
y
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
п
2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/subSubgradients_1/pi/pow_2_grad/sub4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/sub/y*
_output_shapes
: *
T0
й
2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/PowPowpi/sub_22gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/sub*
T0*'
_output_shapes
:         
╒
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_1Mul2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Pow*'
_output_shapes
:         *
T0
ї
2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/SumSum4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_1Dgradients_2/gradients_1/pi/pow_2_grad/Pow_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
ы
6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ReshapeReshape2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Sum4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Shape*
T0*'
_output_shapes
:         *
Tshape0
}
8gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╖
6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/GreaterGreaterpi/sub_28gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Greater/y*
T0*'
_output_shapes
:         
Ж
>gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_like/ShapeShapepi/sub_2*
out_type0*
_output_shapes
:*
T0
Г
>gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Д
8gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_likeFill>gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_like/Shape>gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_like/Const*
T0*

index_type0*'
_output_shapes
:         
э
5gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/SelectSelect6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Greaterpi/sub_28gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_like*
T0*'
_output_shapes
:         
в
2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/LogLog5gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Select*'
_output_shapes
:         *
T0
В
9gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/zeros_like	ZerosLikepi/sub_2*
T0*'
_output_shapes
:         
Ъ
7gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Select_1Select6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Greater2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Log9gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/zeros_like*'
_output_shapes
:         *
T0
╚
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_2Mul:gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Reshape_1gradients_1/pi/pow_2_grad/Pow*'
_output_shapes
:         *
T0
▄
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_3Mul4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_27gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Select_1*
T0*'
_output_shapes
:         
∙
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Sum_1Sum4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_3Fgradients_2/gradients_1/pi/pow_2_grad/Pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
р
8gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Reshape_1Reshape4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Sum_16gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
gradients_2/pi/sub_2_grad/ShapeShapepi/Placeholder*
out_type0*
_output_shapes
:*
T0
s
!gradients_2/pi/sub_2_grad/Shape_1Shapepi/dense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
╔
/gradients_2/pi/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/pi/sub_2_grad/Shape!gradients_2/pi/sub_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
═
gradients_2/pi/sub_2_grad/SumSum6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Reshape/gradients_2/pi/sub_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
м
!gradients_2/pi/sub_2_grad/ReshapeReshapegradients_2/pi/sub_2_grad/Sumgradients_2/pi/sub_2_grad/Shape*
T0*'
_output_shapes
:         *
Tshape0
О
gradients_2/pi/sub_2_grad/NegNeg6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Reshape*
T0*'
_output_shapes
:         
╕
gradients_2/pi/sub_2_grad/Sum_1Sumgradients_2/pi/sub_2_grad/Neg1gradients_2/pi/sub_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
▓
#gradients_2/pi/sub_2_grad/Reshape_1Reshapegradients_2/pi/sub_2_grad/Sum_1!gradients_2/pi/sub_2_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:         
Я
/gradients_2/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_2/pi/sub_2_grad/Reshape_1*
data_formatNHWC*
_output_shapes
:*
T0
╔
)gradients_2/pi/dense_2/MatMul_grad/MatMulMatMul#gradients_2/pi/sub_2_grad/Reshape_1pi/dense_2/kernel/read*(
_output_shapes
:         А*
T0*
transpose_a( *
transpose_b(
╗
+gradients_2/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh#gradients_2/pi/sub_2_grad/Reshape_1*
transpose_a(*
_output_shapes
:	А*
T0*
transpose_b( 
р
gradients_2/AddN_3AddNCgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMul@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul_2)gradients_2/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:         А*V
_classL
JHloc:@gradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMul*
N*
T0
Н
)gradients_2/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanhgradients_2/AddN_3*
T0*(
_output_shapes
:         А
Ч
gradients_2/AddN_4AddNCgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_grad/MatMul_1+gradients_2/pi/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	А*
N*V
_classL
JHloc:@gradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_grad/MatMul_1*
T0
ж
/gradients_2/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_2/pi/dense_1/Tanh_grad/TanhGrad*
data_formatNHWC*
T0*
_output_shapes	
:А
╧
)gradients_2/pi/dense_1/MatMul_grad/MatMulMatMul)gradients_2/pi/dense_1/Tanh_grad/TanhGradpi/dense_1/kernel/read*
T0*
transpose_b(*(
_output_shapes
:         А*
transpose_a( 
└
+gradients_2/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh)gradients_2/pi/dense_1/Tanh_grad/TanhGrad* 
_output_shapes
:
АА*
transpose_a(*
T0*
transpose_b( 
▐
gradients_2/AddN_5AddNCgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMul>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul_2)gradients_2/pi/dense_1/MatMul_grad/MatMul*(
_output_shapes
:         А*
T0*V
_classL
JHloc:@gradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMul*
N
Й
'gradients_2/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanhgradients_2/AddN_5*
T0*(
_output_shapes
:         А
Ш
gradients_2/AddN_6AddNCgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_grad/MatMul_1+gradients_2/pi/dense_1/MatMul_grad/MatMul_1*V
_classL
JHloc:@gradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_grad/MatMul_1*
T0*
N* 
_output_shapes
:
АА
в
-gradients_2/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_2/pi/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
T0*
_output_shapes	
:А
╚
'gradients_2/pi/dense/MatMul_grad/MatMulMatMul'gradients_2/pi/dense/Tanh_grad/TanhGradpi/dense/kernel/read*
transpose_b(*
T0*'
_output_shapes
:         <*
transpose_a( 
╣
)gradients_2/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder'gradients_2/pi/dense/Tanh_grad/TanhGrad*
transpose_b( *
_output_shapes
:	<А*
T0*
transpose_a(
c
Reshape_14/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
Ж

Reshape_14Reshape)gradients_2/pi/dense/MatMul_grad/MatMul_1Reshape_14/shape*
Tshape0*
_output_shapes	
:Аx*
T0
c
Reshape_15/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
К

Reshape_15Reshape-gradients_2/pi/dense/BiasAdd_grad/BiasAddGradReshape_15/shape*
Tshape0*
_output_shapes	
:А*
T0
c
Reshape_16/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
p

Reshape_16Reshapegradients_2/AddN_6Reshape_16/shape*
T0*
_output_shapes

:АА*
Tshape0
c
Reshape_17/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
М

Reshape_17Reshape/gradients_2/pi/dense_1/BiasAdd_grad/BiasAddGradReshape_17/shape*
Tshape0*
_output_shapes	
:А*
T0
c
Reshape_18/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
o

Reshape_18Reshapegradients_2/AddN_4Reshape_18/shape*
T0*
_output_shapes	
:А*
Tshape0
c
Reshape_19/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
Л

Reshape_19Reshape/gradients_2/pi/dense_2/BiasAdd_grad/BiasAddGradReshape_19/shape*
T0*
_output_shapes
:*
Tshape0
c
Reshape_20/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
{

Reshape_20Reshapegradients_2/pi/mul_5_grad/Mul_1Reshape_20/shape*
T0*
_output_shapes
:*
Tshape0
O
concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
│
concat_2ConcatV2
Reshape_14
Reshape_15
Reshape_16
Reshape_17
Reshape_18
Reshape_19
Reshape_20concat_2/axis*
T0*
N*
_output_shapes

:ДА*

Tidx0
L
mul_4/xConst*
dtype0*
valueB
 *═╠╠=*
_output_shapes
: 
K
mul_4Mulmul_4/xPlaceholder_9*
_output_shapes

:ДА*
T0
F
add_1AddV2concat_2mul_4*
T0*
_output_shapes

:ДА
T
gradients_3/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
Z
gradients_3/grad_ys_0Const*
_output_shapes
: *
dtype0*
valueB
 *  А?
u
gradients_3/FillFillgradients_3/Shapegradients_3/grad_ys_0*
T0*
_output_shapes
: *

index_type0
o
%gradients_3/Mean_2_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Ц
gradients_3/Mean_2_grad/ReshapeReshapegradients_3/Fill%gradients_3/Mean_2_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
b
gradients_3/Mean_2_grad/ShapeShapemul_1*
out_type0*
_output_shapes
:*
T0
д
gradients_3/Mean_2_grad/TileTilegradients_3/Mean_2_grad/Reshapegradients_3/Mean_2_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:         
d
gradients_3/Mean_2_grad/Shape_1Shapemul_1*
out_type0*
T0*
_output_shapes
:
b
gradients_3/Mean_2_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
g
gradients_3/Mean_2_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
в
gradients_3/Mean_2_grad/ProdProdgradients_3/Mean_2_grad/Shape_1gradients_3/Mean_2_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
i
gradients_3/Mean_2_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
ж
gradients_3/Mean_2_grad/Prod_1Prodgradients_3/Mean_2_grad/Shape_2gradients_3/Mean_2_grad/Const_1*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
c
!gradients_3/Mean_2_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
О
gradients_3/Mean_2_grad/MaximumMaximumgradients_3/Mean_2_grad/Prod_1!gradients_3/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 
М
 gradients_3/Mean_2_grad/floordivFloorDivgradients_3/Mean_2_grad/Prodgradients_3/Mean_2_grad/Maximum*
_output_shapes
: *
T0
Ж
gradients_3/Mean_2_grad/CastCast gradients_3/Mean_2_grad/floordiv*

DstT0*
Truncate( *
_output_shapes
: *

SrcT0
Ф
gradients_3/Mean_2_grad/truedivRealDivgradients_3/Mean_2_grad/Tilegradients_3/Mean_2_grad/Cast*
T0*#
_output_shapes
:         
_
gradients_3/mul_1_grad/ShapeShapeExp*
T0*
_output_shapes
:*
out_type0
k
gradients_3/mul_1_grad/Shape_1ShapePlaceholder_3*
_output_shapes
:*
out_type0*
T0
└
,gradients_3/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/mul_1_grad/Shapegradients_3/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0

gradients_3/mul_1_grad/MulMulgradients_3/Mean_2_grad/truedivPlaceholder_3*
T0*#
_output_shapes
:         
л
gradients_3/mul_1_grad/SumSumgradients_3/mul_1_grad/Mul,gradients_3/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Я
gradients_3/mul_1_grad/ReshapeReshapegradients_3/mul_1_grad/Sumgradients_3/mul_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
w
gradients_3/mul_1_grad/Mul_1MulExpgradients_3/Mean_2_grad/truediv*#
_output_shapes
:         *
T0
▒
gradients_3/mul_1_grad/Sum_1Sumgradients_3/mul_1_grad/Mul_1.gradients_3/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
е
 gradients_3/mul_1_grad/Reshape_1Reshapegradients_3/mul_1_grad/Sum_1gradients_3/mul_1_grad/Shape_1*
T0*#
_output_shapes
:         *
Tshape0
r
gradients_3/Exp_grad/mulMulgradients_3/mul_1_grad/ReshapeExp*
T0*#
_output_shapes
:         
`
gradients_3/sub_grad/ShapeShapepi/Sum*
out_type0*
T0*
_output_shapes
:
i
gradients_3/sub_grad/Shape_1ShapePlaceholder_6*
out_type0*
T0*
_output_shapes
:
║
*gradients_3/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/sub_grad/Shapegradients_3/sub_grad/Shape_1*2
_output_shapes 
:         :         *
T0
е
gradients_3/sub_grad/SumSumgradients_3/Exp_grad/mul*gradients_3/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Щ
gradients_3/sub_grad/ReshapeReshapegradients_3/sub_grad/Sumgradients_3/sub_grad/Shape*#
_output_shapes
:         *
Tshape0*
T0
g
gradients_3/sub_grad/NegNeggradients_3/Exp_grad/mul*
T0*#
_output_shapes
:         
й
gradients_3/sub_grad/Sum_1Sumgradients_3/sub_grad/Neg,gradients_3/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Я
gradients_3/sub_grad/Reshape_1Reshapegradients_3/sub_grad/Sum_1gradients_3/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:         
e
gradients_3/pi/Sum_grad/ShapeShapepi/mul_2*
_output_shapes
:*
out_type0*
T0
Р
gradients_3/pi/Sum_grad/SizeConst*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
dtype0*
value	B :*
_output_shapes
: 
п
gradients_3/pi/Sum_grad/addAddV2pi/Sum/reduction_indicesgradients_3/pi/Sum_grad/Size*
T0*
_output_shapes
: *0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape
╡
gradients_3/pi/Sum_grad/modFloorModgradients_3/pi/Sum_grad/addgradients_3/pi/Sum_grad/Size*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
_output_shapes
: *
T0
Ф
gradients_3/pi/Sum_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: *0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape
Ч
#gradients_3/pi/Sum_grad/range/startConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
value	B : 
Ч
#gradients_3/pi/Sum_grad/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape
ш
gradients_3/pi/Sum_grad/rangeRange#gradients_3/pi/Sum_grad/range/startgradients_3/pi/Sum_grad/Size#gradients_3/pi/Sum_grad/range/delta*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*

Tidx0*
_output_shapes
:
Ц
"gradients_3/pi/Sum_grad/Fill/valueConst*
value	B :*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
_output_shapes
: *
dtype0
╬
gradients_3/pi/Sum_grad/FillFillgradients_3/pi/Sum_grad/Shape_1"gradients_3/pi/Sum_grad/Fill/value*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
_output_shapes
: *

index_type0*
T0
П
%gradients_3/pi/Sum_grad/DynamicStitchDynamicStitchgradients_3/pi/Sum_grad/rangegradients_3/pi/Sum_grad/modgradients_3/pi/Sum_grad/Shapegradients_3/pi/Sum_grad/Fill*
N*
_output_shapes
:*
T0*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape
Х
!gradients_3/pi/Sum_grad/Maximum/yConst*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
_output_shapes
: *
value	B :*
dtype0
╦
gradients_3/pi/Sum_grad/MaximumMaximum%gradients_3/pi/Sum_grad/DynamicStitch!gradients_3/pi/Sum_grad/Maximum/y*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
T0*
_output_shapes
:
├
 gradients_3/pi/Sum_grad/floordivFloorDivgradients_3/pi/Sum_grad/Shapegradients_3/pi/Sum_grad/Maximum*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
_output_shapes
:*
T0
╕
gradients_3/pi/Sum_grad/ReshapeReshapegradients_3/sub_grad/Reshape%gradients_3/pi/Sum_grad/DynamicStitch*
Tshape0*0
_output_shapes
:                  *
T0
л
gradients_3/pi/Sum_grad/TileTilegradients_3/pi/Sum_grad/Reshape gradients_3/pi/Sum_grad/floordiv*
T0*

Tmultiples0*'
_output_shapes
:         
g
gradients_3/pi/mul_2_grad/ShapeShape
pi/mul_2/x*
out_type0*
_output_shapes
: *
T0
i
!gradients_3/pi/mul_2_grad/Shape_1Shapepi/add_3*
T0*
out_type0*
_output_shapes
:
╔
/gradients_3/pi/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/mul_2_grad/Shape!gradients_3/pi/mul_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
~
gradients_3/pi/mul_2_grad/MulMulgradients_3/pi/Sum_grad/Tilepi/add_3*
T0*'
_output_shapes
:         
┤
gradients_3/pi/mul_2_grad/SumSumgradients_3/pi/mul_2_grad/Mul/gradients_3/pi/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ы
!gradients_3/pi/mul_2_grad/ReshapeReshapegradients_3/pi/mul_2_grad/Sumgradients_3/pi/mul_2_grad/Shape*
_output_shapes
: *
Tshape0*
T0
В
gradients_3/pi/mul_2_grad/Mul_1Mul
pi/mul_2/xgradients_3/pi/Sum_grad/Tile*'
_output_shapes
:         *
T0
║
gradients_3/pi/mul_2_grad/Sum_1Sumgradients_3/pi/mul_2_grad/Mul_11gradients_3/pi/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
▓
#gradients_3/pi/mul_2_grad/Reshape_1Reshapegradients_3/pi/mul_2_grad/Sum_1!gradients_3/pi/mul_2_grad/Shape_1*
T0*'
_output_shapes
:         *
Tshape0
g
gradients_3/pi/add_3_grad/ShapeShapepi/add_2*
out_type0*
T0*
_output_shapes
:
i
!gradients_3/pi/add_3_grad/Shape_1Shape
pi/add_3/y*
T0*
out_type0*
_output_shapes
: 
╔
/gradients_3/pi/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/add_3_grad/Shape!gradients_3/pi/add_3_grad/Shape_1*
T0*2
_output_shapes 
:         :         
║
gradients_3/pi/add_3_grad/SumSum#gradients_3/pi/mul_2_grad/Reshape_1/gradients_3/pi/add_3_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:*

Tidx0
м
!gradients_3/pi/add_3_grad/ReshapeReshapegradients_3/pi/add_3_grad/Sumgradients_3/pi/add_3_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
╛
gradients_3/pi/add_3_grad/Sum_1Sum#gradients_3/pi/mul_2_grad/Reshape_11gradients_3/pi/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
б
#gradients_3/pi/add_3_grad/Reshape_1Reshapegradients_3/pi/add_3_grad/Sum_1!gradients_3/pi/add_3_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
e
gradients_3/pi/add_2_grad/ShapeShapepi/pow*
_output_shapes
:*
out_type0*
T0
i
!gradients_3/pi/add_2_grad/Shape_1Shapepi/mul_1*
T0*
out_type0*
_output_shapes
:
╔
/gradients_3/pi/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/add_2_grad/Shape!gradients_3/pi/add_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╕
gradients_3/pi/add_2_grad/SumSum!gradients_3/pi/add_3_grad/Reshape/gradients_3/pi/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
м
!gradients_3/pi/add_2_grad/ReshapeReshapegradients_3/pi/add_2_grad/Sumgradients_3/pi/add_2_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
╝
gradients_3/pi/add_2_grad/Sum_1Sum!gradients_3/pi/add_3_grad/Reshape1gradients_3/pi/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
е
#gradients_3/pi/add_2_grad/Reshape_1Reshapegradients_3/pi/add_2_grad/Sum_1!gradients_3/pi/add_2_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
g
gradients_3/pi/pow_grad/ShapeShape
pi/truediv*
T0*
out_type0*
_output_shapes
:
e
gradients_3/pi/pow_grad/Shape_1Shapepi/pow/y*
T0*
out_type0*
_output_shapes
: 
├
-gradients_3/pi/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/pow_grad/Shapegradients_3/pi/pow_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Б
gradients_3/pi/pow_grad/mulMul!gradients_3/pi/add_2_grad/Reshapepi/pow/y*'
_output_shapes
:         *
T0
b
gradients_3/pi/pow_grad/sub/yConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
l
gradients_3/pi/pow_grad/subSubpi/pow/ygradients_3/pi/pow_grad/sub/y*
T0*
_output_shapes
: 
}
gradients_3/pi/pow_grad/PowPow
pi/truedivgradients_3/pi/pow_grad/sub*'
_output_shapes
:         *
T0
Р
gradients_3/pi/pow_grad/mul_1Mulgradients_3/pi/pow_grad/mulgradients_3/pi/pow_grad/Pow*'
_output_shapes
:         *
T0
░
gradients_3/pi/pow_grad/SumSumgradients_3/pi/pow_grad/mul_1-gradients_3/pi/pow_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
ж
gradients_3/pi/pow_grad/ReshapeReshapegradients_3/pi/pow_grad/Sumgradients_3/pi/pow_grad/Shape*'
_output_shapes
:         *
Tshape0*
T0
f
!gradients_3/pi/pow_grad/Greater/yConst*
valueB
 *    *
_output_shapes
: *
dtype0
Л
gradients_3/pi/pow_grad/GreaterGreater
pi/truediv!gradients_3/pi/pow_grad/Greater/y*'
_output_shapes
:         *
T0
q
'gradients_3/pi/pow_grad/ones_like/ShapeShape
pi/truediv*
_output_shapes
:*
out_type0*
T0
l
'gradients_3/pi/pow_grad/ones_like/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
┐
!gradients_3/pi/pow_grad/ones_likeFill'gradients_3/pi/pow_grad/ones_like/Shape'gradients_3/pi/pow_grad/ones_like/Const*

index_type0*'
_output_shapes
:         *
T0
к
gradients_3/pi/pow_grad/SelectSelectgradients_3/pi/pow_grad/Greater
pi/truediv!gradients_3/pi/pow_grad/ones_like*
T0*'
_output_shapes
:         
t
gradients_3/pi/pow_grad/LogLoggradients_3/pi/pow_grad/Select*'
_output_shapes
:         *
T0
m
"gradients_3/pi/pow_grad/zeros_like	ZerosLike
pi/truediv*
T0*'
_output_shapes
:         
╛
 gradients_3/pi/pow_grad/Select_1Selectgradients_3/pi/pow_grad/Greatergradients_3/pi/pow_grad/Log"gradients_3/pi/pow_grad/zeros_like*
T0*'
_output_shapes
:         
Б
gradients_3/pi/pow_grad/mul_2Mul!gradients_3/pi/add_2_grad/Reshapepi/pow*'
_output_shapes
:         *
T0
Ч
gradients_3/pi/pow_grad/mul_3Mulgradients_3/pi/pow_grad/mul_2 gradients_3/pi/pow_grad/Select_1*
T0*'
_output_shapes
:         
┤
gradients_3/pi/pow_grad/Sum_1Sumgradients_3/pi/pow_grad/mul_3/gradients_3/pi/pow_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ы
!gradients_3/pi/pow_grad/Reshape_1Reshapegradients_3/pi/pow_grad/Sum_1gradients_3/pi/pow_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0

gradients_3/pi/mul_1_grad/MulMul#gradients_3/pi/add_2_grad/Reshape_1pi/log_std/read*
T0*
_output_shapes
:
y
/gradients_3/pi/mul_1_grad/Sum/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0
▓
gradients_3/pi/mul_1_grad/SumSumgradients_3/pi/mul_1_grad/Mul/gradients_3/pi/mul_1_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
j
'gradients_3/pi/mul_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
г
!gradients_3/pi/mul_1_grad/ReshapeReshapegradients_3/pi/mul_1_grad/Sum'gradients_3/pi/mul_1_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
: 
|
gradients_3/pi/mul_1_grad/Mul_1Mul
pi/mul_1/x#gradients_3/pi/add_2_grad/Reshape_1*
T0*
_output_shapes
:
g
!gradients_3/pi/truediv_grad/ShapeShapepi/sub*
_output_shapes
:*
out_type0*
T0
m
#gradients_3/pi/truediv_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
╧
1gradients_3/pi/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients_3/pi/truediv_grad/Shape#gradients_3/pi/truediv_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Л
#gradients_3/pi/truediv_grad/RealDivRealDivgradients_3/pi/pow_grad/Reshapepi/add_1*
T0*'
_output_shapes
:         
╛
gradients_3/pi/truediv_grad/SumSum#gradients_3/pi/truediv_grad/RealDiv1gradients_3/pi/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
▓
#gradients_3/pi/truediv_grad/ReshapeReshapegradients_3/pi/truediv_grad/Sum!gradients_3/pi/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
`
gradients_3/pi/truediv_grad/NegNegpi/sub*
T0*'
_output_shapes
:         
Н
%gradients_3/pi/truediv_grad/RealDiv_1RealDivgradients_3/pi/truediv_grad/Negpi/add_1*
T0*'
_output_shapes
:         
У
%gradients_3/pi/truediv_grad/RealDiv_2RealDiv%gradients_3/pi/truediv_grad/RealDiv_1pi/add_1*'
_output_shapes
:         *
T0
а
gradients_3/pi/truediv_grad/mulMulgradients_3/pi/pow_grad/Reshape%gradients_3/pi/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:         
╛
!gradients_3/pi/truediv_grad/Sum_1Sumgradients_3/pi/truediv_grad/mul3gradients_3/pi/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
л
%gradients_3/pi/truediv_grad/Reshape_1Reshape!gradients_3/pi/truediv_grad/Sum_1#gradients_3/pi/truediv_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
j
gradients_3/pi/sub_grad/ShapeShapePlaceholder_1*
T0*
_output_shapes
:*
out_type0
q
gradients_3/pi/sub_grad/Shape_1Shapepi/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
├
-gradients_3/pi/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/sub_grad/Shapegradients_3/pi/sub_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╢
gradients_3/pi/sub_grad/SumSum#gradients_3/pi/truediv_grad/Reshape-gradients_3/pi/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
ж
gradients_3/pi/sub_grad/ReshapeReshapegradients_3/pi/sub_grad/Sumgradients_3/pi/sub_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
y
gradients_3/pi/sub_grad/NegNeg#gradients_3/pi/truediv_grad/Reshape*'
_output_shapes
:         *
T0
▓
gradients_3/pi/sub_grad/Sum_1Sumgradients_3/pi/sub_grad/Neg/gradients_3/pi/sub_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
м
!gradients_3/pi/sub_grad/Reshape_1Reshapegradients_3/pi/sub_grad/Sum_1gradients_3/pi/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
y
/gradients_3/pi/add_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
valueB: *
dtype0
║
gradients_3/pi/add_1_grad/SumSum%gradients_3/pi/truediv_grad/Reshape_1/gradients_3/pi/add_1_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
j
'gradients_3/pi/add_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
г
!gradients_3/pi/add_1_grad/ReshapeReshapegradients_3/pi/add_1_grad/Sum'gradients_3/pi/add_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
Э
/gradients_3/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad!gradients_3/pi/sub_grad/Reshape_1*
T0*
data_formatNHWC*
_output_shapes
:
z
gradients_3/pi/Exp_1_grad/mulMul%gradients_3/pi/truediv_grad/Reshape_1pi/Exp_1*
_output_shapes
:*
T0
╟
)gradients_3/pi/dense_2/MatMul_grad/MatMulMatMul!gradients_3/pi/sub_grad/Reshape_1pi/dense_2/kernel/read*
transpose_b(*
transpose_a( *(
_output_shapes
:         А*
T0
╣
+gradients_3/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh!gradients_3/pi/sub_grad/Reshape_1*
_output_shapes
:	А*
transpose_b( *
T0*
transpose_a(
║
gradients_3/AddNAddNgradients_3/pi/mul_1_grad/Mul_1gradients_3/pi/Exp_1_grad/mul*
_output_shapes
:*
T0*
N*2
_class(
&$loc:@gradients_3/pi/mul_1_grad/Mul_1
д
)gradients_3/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh)gradients_3/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:         А*
T0
ж
/gradients_3/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_3/pi/dense_1/Tanh_grad/TanhGrad*
T0*
_output_shapes	
:А*
data_formatNHWC
╧
)gradients_3/pi/dense_1/MatMul_grad/MatMulMatMul)gradients_3/pi/dense_1/Tanh_grad/TanhGradpi/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b(*(
_output_shapes
:         А
└
+gradients_3/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh)gradients_3/pi/dense_1/Tanh_grad/TanhGrad*
transpose_a(*
transpose_b( * 
_output_shapes
:
АА*
T0
а
'gradients_3/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh)gradients_3/pi/dense_1/MatMul_grad/MatMul*
T0*(
_output_shapes
:         А
в
-gradients_3/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_3/pi/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
T0*
_output_shapes	
:А
╚
'gradients_3/pi/dense/MatMul_grad/MatMulMatMul'gradients_3/pi/dense/Tanh_grad/TanhGradpi/dense/kernel/read*
transpose_a( *
T0*'
_output_shapes
:         <*
transpose_b(
╣
)gradients_3/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder'gradients_3/pi/dense/Tanh_grad/TanhGrad*
_output_shapes
:	<А*
T0*
transpose_a(*
transpose_b( 
c
Reshape_21/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
Ж

Reshape_21Reshape)gradients_3/pi/dense/MatMul_grad/MatMul_1Reshape_21/shape*
_output_shapes	
:Аx*
T0*
Tshape0
c
Reshape_22/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
К

Reshape_22Reshape-gradients_3/pi/dense/BiasAdd_grad/BiasAddGradReshape_22/shape*
T0*
_output_shapes	
:А*
Tshape0
c
Reshape_23/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
Й

Reshape_23Reshape+gradients_3/pi/dense_1/MatMul_grad/MatMul_1Reshape_23/shape*
_output_shapes

:АА*
Tshape0*
T0
c
Reshape_24/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
М

Reshape_24Reshape/gradients_3/pi/dense_1/BiasAdd_grad/BiasAddGradReshape_24/shape*
Tshape0*
T0*
_output_shapes	
:А
c
Reshape_25/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
И

Reshape_25Reshape+gradients_3/pi/dense_2/MatMul_grad/MatMul_1Reshape_25/shape*
T0*
_output_shapes	
:А*
Tshape0
c
Reshape_26/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
Л

Reshape_26Reshape/gradients_3/pi/dense_2/BiasAdd_grad/BiasAddGradReshape_26/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_27/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
l

Reshape_27Reshapegradients_3/AddNReshape_27/shape*
T0*
Tshape0*
_output_shapes
:
O
concat_3/axisConst*
value	B : *
dtype0*
_output_shapes
: 
│
concat_3ConcatV2
Reshape_21
Reshape_22
Reshape_23
Reshape_24
Reshape_25
Reshape_26
Reshape_27concat_3/axis*
T0*
_output_shapes

:ДА*

Tidx0*
N
c
Reshape_28/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
q

Reshape_28Reshapepi/dense/kernel/readReshape_28/shape*
_output_shapes	
:Аx*
T0*
Tshape0
c
Reshape_29/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
o

Reshape_29Reshapepi/dense/bias/readReshape_29/shape*
Tshape0*
T0*
_output_shapes	
:А
c
Reshape_30/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
t

Reshape_30Reshapepi/dense_1/kernel/readReshape_30/shape*
_output_shapes

:АА*
Tshape0*
T0
c
Reshape_31/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
q

Reshape_31Reshapepi/dense_1/bias/readReshape_31/shape*
Tshape0*
T0*
_output_shapes	
:А
c
Reshape_32/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
s

Reshape_32Reshapepi/dense_2/kernel/readReshape_32/shape*
T0*
_output_shapes	
:А*
Tshape0
c
Reshape_33/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
p

Reshape_33Reshapepi/dense_2/bias/readReshape_33/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_34/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
k

Reshape_34Reshapepi/log_std/readReshape_34/shape*
_output_shapes
:*
Tshape0*
T0
O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
│
concat_4ConcatV2
Reshape_28
Reshape_29
Reshape_30
Reshape_31
Reshape_32
Reshape_33
Reshape_34concat_4/axis*
_output_shapes

:ДА*
N*
T0*

Tidx0
l
Const_4Const*
dtype0*1
value(B&" <                    *
_output_shapes
:
Q
split/split_dimConst*
_output_shapes
: *
value	B : *
dtype0
д
splitSplitVPlaceholder_9Const_4split/split_dim*D
_output_shapes2
0:Аx:А:АА:А:А::*

Tlen0*
T0*
	num_split
a
Reshape_35/shapeConst*
_output_shapes
:*
valueB"<      *
dtype0
f

Reshape_35ReshapesplitReshape_35/shape*
Tshape0*
_output_shapes
:	<А*
T0
[
Reshape_36/shapeConst*
dtype0*
valueB:А*
_output_shapes
:
d

Reshape_36Reshapesplit:1Reshape_36/shape*
T0*
Tshape0*
_output_shapes	
:А
a
Reshape_37/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
i

Reshape_37Reshapesplit:2Reshape_37/shape*
Tshape0* 
_output_shapes
:
АА*
T0
[
Reshape_38/shapeConst*
dtype0*
_output_shapes
:*
valueB:А
d

Reshape_38Reshapesplit:3Reshape_38/shape*
Tshape0*
T0*
_output_shapes	
:А
a
Reshape_39/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
h

Reshape_39Reshapesplit:4Reshape_39/shape*
_output_shapes
:	А*
T0*
Tshape0
Z
Reshape_40/shapeConst*
valueB:*
_output_shapes
:*
dtype0
c

Reshape_40Reshapesplit:5Reshape_40/shape*
T0*
Tshape0*
_output_shapes
:
Z
Reshape_41/shapeConst*
_output_shapes
:*
valueB:*
dtype0
c

Reshape_41Reshapesplit:6Reshape_41/shape*
T0*
Tshape0*
_output_shapes
:
д
AssignAssignpi/dense/kernel
Reshape_35*
validate_shape(*
use_locking(*
_output_shapes
:	<А*"
_class
loc:@pi/dense/kernel*
T0
Ю
Assign_1Assignpi/dense/bias
Reshape_36*
_output_shapes	
:А*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
use_locking(
л
Assign_2Assignpi/dense_1/kernel
Reshape_37*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
АА*
use_locking(
в
Assign_3Assignpi/dense_1/bias
Reshape_38*
use_locking(*
validate_shape(*
_output_shapes	
:А*"
_class
loc:@pi/dense_1/bias*
T0
к
Assign_4Assignpi/dense_2/kernel
Reshape_39*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	А*$
_class
loc:@pi/dense_2/kernel
б
Assign_5Assignpi/dense_2/bias
Reshape_40*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(
Ч
Assign_6Assign
pi/log_std
Reshape_41*
_output_shapes
:*
validate_shape(*
T0*
_class
loc:@pi/log_std*
use_locking(
]

group_depsNoOp^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6
U
sub_1SubPlaceholder_4
vf/Squeeze*
T0*#
_output_shapes
:         
J
pow/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0
F
powPowsub_1pow/y*#
_output_shapes
:         *
T0
Q
Const_5Const*
_output_shapes
:*
valueB: *
dtype0
Z
Mean_3MeanpowConst_5*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
U
sub_2SubPlaceholder_5
vc/Squeeze*#
_output_shapes
:         *
T0
L
pow_1/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0
J
pow_1Powsub_2pow_1/y*#
_output_shapes
:         *
T0
Q
Const_6Const*
_output_shapes
:*
dtype0*
valueB: 
\
Mean_4Meanpow_1Const_6*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
?
add_2AddV2Mean_3Mean_4*
_output_shapes
: *
T0
T
gradients_4/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
Z
gradients_4/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  А?
u
gradients_4/FillFillgradients_4/Shapegradients_4/grad_ys_0*
_output_shapes
: *
T0*

index_type0
B
'gradients_4/add_2_grad/tuple/group_depsNoOp^gradients_4/Fill
╜
/gradients_4/add_2_grad/tuple/control_dependencyIdentitygradients_4/Fill(^gradients_4/add_2_grad/tuple/group_deps*
T0*
_output_shapes
: *#
_class
loc:@gradients_4/Fill
┐
1gradients_4/add_2_grad/tuple/control_dependency_1Identitygradients_4/Fill(^gradients_4/add_2_grad/tuple/group_deps*#
_class
loc:@gradients_4/Fill*
_output_shapes
: *
T0
o
%gradients_4/Mean_3_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
╡
gradients_4/Mean_3_grad/ReshapeReshape/gradients_4/add_2_grad/tuple/control_dependency%gradients_4/Mean_3_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
`
gradients_4/Mean_3_grad/ShapeShapepow*
_output_shapes
:*
T0*
out_type0
д
gradients_4/Mean_3_grad/TileTilegradients_4/Mean_3_grad/Reshapegradients_4/Mean_3_grad/Shape*
T0*#
_output_shapes
:         *

Tmultiples0
b
gradients_4/Mean_3_grad/Shape_1Shapepow*
T0*
out_type0*
_output_shapes
:
b
gradients_4/Mean_3_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
g
gradients_4/Mean_3_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
в
gradients_4/Mean_3_grad/ProdProdgradients_4/Mean_3_grad/Shape_1gradients_4/Mean_3_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
i
gradients_4/Mean_3_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
ж
gradients_4/Mean_3_grad/Prod_1Prodgradients_4/Mean_3_grad/Shape_2gradients_4/Mean_3_grad/Const_1*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
c
!gradients_4/Mean_3_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
О
gradients_4/Mean_3_grad/MaximumMaximumgradients_4/Mean_3_grad/Prod_1!gradients_4/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 
М
 gradients_4/Mean_3_grad/floordivFloorDivgradients_4/Mean_3_grad/Prodgradients_4/Mean_3_grad/Maximum*
_output_shapes
: *
T0
Ж
gradients_4/Mean_3_grad/CastCast gradients_4/Mean_3_grad/floordiv*

DstT0*
Truncate( *
_output_shapes
: *

SrcT0
Ф
gradients_4/Mean_3_grad/truedivRealDivgradients_4/Mean_3_grad/Tilegradients_4/Mean_3_grad/Cast*
T0*#
_output_shapes
:         
o
%gradients_4/Mean_4_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
╖
gradients_4/Mean_4_grad/ReshapeReshape1gradients_4/add_2_grad/tuple/control_dependency_1%gradients_4/Mean_4_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
b
gradients_4/Mean_4_grad/ShapeShapepow_1*
_output_shapes
:*
out_type0*
T0
д
gradients_4/Mean_4_grad/TileTilegradients_4/Mean_4_grad/Reshapegradients_4/Mean_4_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:         
d
gradients_4/Mean_4_grad/Shape_1Shapepow_1*
T0*
_output_shapes
:*
out_type0
b
gradients_4/Mean_4_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
g
gradients_4/Mean_4_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
в
gradients_4/Mean_4_grad/ProdProdgradients_4/Mean_4_grad/Shape_1gradients_4/Mean_4_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_4/Mean_4_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
ж
gradients_4/Mean_4_grad/Prod_1Prodgradients_4/Mean_4_grad/Shape_2gradients_4/Mean_4_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_4/Mean_4_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
О
gradients_4/Mean_4_grad/MaximumMaximumgradients_4/Mean_4_grad/Prod_1!gradients_4/Mean_4_grad/Maximum/y*
T0*
_output_shapes
: 
М
 gradients_4/Mean_4_grad/floordivFloorDivgradients_4/Mean_4_grad/Prodgradients_4/Mean_4_grad/Maximum*
_output_shapes
: *
T0
Ж
gradients_4/Mean_4_grad/CastCast gradients_4/Mean_4_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
Ф
gradients_4/Mean_4_grad/truedivRealDivgradients_4/Mean_4_grad/Tilegradients_4/Mean_4_grad/Cast*
T0*#
_output_shapes
:         
_
gradients_4/pow_grad/ShapeShapesub_1*
out_type0*
T0*
_output_shapes
:
_
gradients_4/pow_grad/Shape_1Shapepow/y*
T0*
out_type0*
_output_shapes
: 
║
*gradients_4/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/pow_grad/Shapegradients_4/pow_grad/Shape_1*2
_output_shapes 
:         :         *
T0
u
gradients_4/pow_grad/mulMulgradients_4/Mean_3_grad/truedivpow/y*
T0*#
_output_shapes
:         
_
gradients_4/pow_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
c
gradients_4/pow_grad/subSubpow/ygradients_4/pow_grad/sub/y*
_output_shapes
: *
T0
n
gradients_4/pow_grad/PowPowsub_1gradients_4/pow_grad/sub*
T0*#
_output_shapes
:         
Г
gradients_4/pow_grad/mul_1Mulgradients_4/pow_grad/mulgradients_4/pow_grad/Pow*#
_output_shapes
:         *
T0
з
gradients_4/pow_grad/SumSumgradients_4/pow_grad/mul_1*gradients_4/pow_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Щ
gradients_4/pow_grad/ReshapeReshapegradients_4/pow_grad/Sumgradients_4/pow_grad/Shape*
Tshape0*
T0*#
_output_shapes
:         
c
gradients_4/pow_grad/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
|
gradients_4/pow_grad/GreaterGreatersub_1gradients_4/pow_grad/Greater/y*#
_output_shapes
:         *
T0
i
$gradients_4/pow_grad/ones_like/ShapeShapesub_1*
T0*
_output_shapes
:*
out_type0
i
$gradients_4/pow_grad/ones_like/ConstConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
▓
gradients_4/pow_grad/ones_likeFill$gradients_4/pow_grad/ones_like/Shape$gradients_4/pow_grad/ones_like/Const*
T0*#
_output_shapes
:         *

index_type0
Ш
gradients_4/pow_grad/SelectSelectgradients_4/pow_grad/Greatersub_1gradients_4/pow_grad/ones_like*
T0*#
_output_shapes
:         
j
gradients_4/pow_grad/LogLoggradients_4/pow_grad/Select*
T0*#
_output_shapes
:         
a
gradients_4/pow_grad/zeros_like	ZerosLikesub_1*
T0*#
_output_shapes
:         
о
gradients_4/pow_grad/Select_1Selectgradients_4/pow_grad/Greatergradients_4/pow_grad/Loggradients_4/pow_grad/zeros_like*#
_output_shapes
:         *
T0
u
gradients_4/pow_grad/mul_2Mulgradients_4/Mean_3_grad/truedivpow*#
_output_shapes
:         *
T0
К
gradients_4/pow_grad/mul_3Mulgradients_4/pow_grad/mul_2gradients_4/pow_grad/Select_1*#
_output_shapes
:         *
T0
л
gradients_4/pow_grad/Sum_1Sumgradients_4/pow_grad/mul_3,gradients_4/pow_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
Т
gradients_4/pow_grad/Reshape_1Reshapegradients_4/pow_grad/Sum_1gradients_4/pow_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
m
%gradients_4/pow_grad/tuple/group_depsNoOp^gradients_4/pow_grad/Reshape^gradients_4/pow_grad/Reshape_1
▐
-gradients_4/pow_grad/tuple/control_dependencyIdentitygradients_4/pow_grad/Reshape&^gradients_4/pow_grad/tuple/group_deps*#
_output_shapes
:         *
T0*/
_class%
#!loc:@gradients_4/pow_grad/Reshape
╫
/gradients_4/pow_grad/tuple/control_dependency_1Identitygradients_4/pow_grad/Reshape_1&^gradients_4/pow_grad/tuple/group_deps*
T0*
_output_shapes
: *1
_class'
%#loc:@gradients_4/pow_grad/Reshape_1
a
gradients_4/pow_1_grad/ShapeShapesub_2*
T0*
out_type0*
_output_shapes
:
c
gradients_4/pow_1_grad/Shape_1Shapepow_1/y*
out_type0*
T0*
_output_shapes
: 
└
,gradients_4/pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/pow_1_grad/Shapegradients_4/pow_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
y
gradients_4/pow_1_grad/mulMulgradients_4/Mean_4_grad/truedivpow_1/y*#
_output_shapes
:         *
T0
a
gradients_4/pow_1_grad/sub/yConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
i
gradients_4/pow_1_grad/subSubpow_1/ygradients_4/pow_1_grad/sub/y*
T0*
_output_shapes
: 
r
gradients_4/pow_1_grad/PowPowsub_2gradients_4/pow_1_grad/sub*
T0*#
_output_shapes
:         
Й
gradients_4/pow_1_grad/mul_1Mulgradients_4/pow_1_grad/mulgradients_4/pow_1_grad/Pow*#
_output_shapes
:         *
T0
н
gradients_4/pow_1_grad/SumSumgradients_4/pow_1_grad/mul_1,gradients_4/pow_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
Я
gradients_4/pow_1_grad/ReshapeReshapegradients_4/pow_1_grad/Sumgradients_4/pow_1_grad/Shape*
Tshape0*#
_output_shapes
:         *
T0
e
 gradients_4/pow_1_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
А
gradients_4/pow_1_grad/GreaterGreatersub_2 gradients_4/pow_1_grad/Greater/y*#
_output_shapes
:         *
T0
k
&gradients_4/pow_1_grad/ones_like/ShapeShapesub_2*
_output_shapes
:*
out_type0*
T0
k
&gradients_4/pow_1_grad/ones_like/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╕
 gradients_4/pow_1_grad/ones_likeFill&gradients_4/pow_1_grad/ones_like/Shape&gradients_4/pow_1_grad/ones_like/Const*

index_type0*
T0*#
_output_shapes
:         
Ю
gradients_4/pow_1_grad/SelectSelectgradients_4/pow_1_grad/Greatersub_2 gradients_4/pow_1_grad/ones_like*#
_output_shapes
:         *
T0
n
gradients_4/pow_1_grad/LogLoggradients_4/pow_1_grad/Select*
T0*#
_output_shapes
:         
c
!gradients_4/pow_1_grad/zeros_like	ZerosLikesub_2*
T0*#
_output_shapes
:         
╢
gradients_4/pow_1_grad/Select_1Selectgradients_4/pow_1_grad/Greatergradients_4/pow_1_grad/Log!gradients_4/pow_1_grad/zeros_like*
T0*#
_output_shapes
:         
y
gradients_4/pow_1_grad/mul_2Mulgradients_4/Mean_4_grad/truedivpow_1*#
_output_shapes
:         *
T0
Р
gradients_4/pow_1_grad/mul_3Mulgradients_4/pow_1_grad/mul_2gradients_4/pow_1_grad/Select_1*
T0*#
_output_shapes
:         
▒
gradients_4/pow_1_grad/Sum_1Sumgradients_4/pow_1_grad/mul_3.gradients_4/pow_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ш
 gradients_4/pow_1_grad/Reshape_1Reshapegradients_4/pow_1_grad/Sum_1gradients_4/pow_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_4/pow_1_grad/tuple/group_depsNoOp^gradients_4/pow_1_grad/Reshape!^gradients_4/pow_1_grad/Reshape_1
ц
/gradients_4/pow_1_grad/tuple/control_dependencyIdentitygradients_4/pow_1_grad/Reshape(^gradients_4/pow_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_4/pow_1_grad/Reshape*#
_output_shapes
:         
▀
1gradients_4/pow_1_grad/tuple/control_dependency_1Identity gradients_4/pow_1_grad/Reshape_1(^gradients_4/pow_1_grad/tuple/group_deps*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients_4/pow_1_grad/Reshape_1
i
gradients_4/sub_1_grad/ShapeShapePlaceholder_4*
_output_shapes
:*
out_type0*
T0
h
gradients_4/sub_1_grad/Shape_1Shape
vf/Squeeze*
T0*
_output_shapes
:*
out_type0
└
,gradients_4/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/sub_1_grad/Shapegradients_4/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╛
gradients_4/sub_1_grad/SumSum-gradients_4/pow_grad/tuple/control_dependency,gradients_4/sub_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
Я
gradients_4/sub_1_grad/ReshapeReshapegradients_4/sub_1_grad/Sumgradients_4/sub_1_grad/Shape*#
_output_shapes
:         *
T0*
Tshape0
~
gradients_4/sub_1_grad/NegNeg-gradients_4/pow_grad/tuple/control_dependency*#
_output_shapes
:         *
T0
п
gradients_4/sub_1_grad/Sum_1Sumgradients_4/sub_1_grad/Neg.gradients_4/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
е
 gradients_4/sub_1_grad/Reshape_1Reshapegradients_4/sub_1_grad/Sum_1gradients_4/sub_1_grad/Shape_1*
T0*#
_output_shapes
:         *
Tshape0
s
'gradients_4/sub_1_grad/tuple/group_depsNoOp^gradients_4/sub_1_grad/Reshape!^gradients_4/sub_1_grad/Reshape_1
ц
/gradients_4/sub_1_grad/tuple/control_dependencyIdentitygradients_4/sub_1_grad/Reshape(^gradients_4/sub_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients_4/sub_1_grad/Reshape*
T0*#
_output_shapes
:         
ь
1gradients_4/sub_1_grad/tuple/control_dependency_1Identity gradients_4/sub_1_grad/Reshape_1(^gradients_4/sub_1_grad/tuple/group_deps*#
_output_shapes
:         *
T0*3
_class)
'%loc:@gradients_4/sub_1_grad/Reshape_1
i
gradients_4/sub_2_grad/ShapeShapePlaceholder_5*
out_type0*
_output_shapes
:*
T0
h
gradients_4/sub_2_grad/Shape_1Shape
vc/Squeeze*
out_type0*
T0*
_output_shapes
:
└
,gradients_4/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/sub_2_grad/Shapegradients_4/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
└
gradients_4/sub_2_grad/SumSum/gradients_4/pow_1_grad/tuple/control_dependency,gradients_4/sub_2_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Я
gradients_4/sub_2_grad/ReshapeReshapegradients_4/sub_2_grad/Sumgradients_4/sub_2_grad/Shape*#
_output_shapes
:         *
Tshape0*
T0
А
gradients_4/sub_2_grad/NegNeg/gradients_4/pow_1_grad/tuple/control_dependency*#
_output_shapes
:         *
T0
п
gradients_4/sub_2_grad/Sum_1Sumgradients_4/sub_2_grad/Neg.gradients_4/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
е
 gradients_4/sub_2_grad/Reshape_1Reshapegradients_4/sub_2_grad/Sum_1gradients_4/sub_2_grad/Shape_1*#
_output_shapes
:         *
T0*
Tshape0
s
'gradients_4/sub_2_grad/tuple/group_depsNoOp^gradients_4/sub_2_grad/Reshape!^gradients_4/sub_2_grad/Reshape_1
ц
/gradients_4/sub_2_grad/tuple/control_dependencyIdentitygradients_4/sub_2_grad/Reshape(^gradients_4/sub_2_grad/tuple/group_deps*
T0*#
_output_shapes
:         *1
_class'
%#loc:@gradients_4/sub_2_grad/Reshape
ь
1gradients_4/sub_2_grad/tuple/control_dependency_1Identity gradients_4/sub_2_grad/Reshape_1(^gradients_4/sub_2_grad/tuple/group_deps*#
_output_shapes
:         *
T0*3
_class)
'%loc:@gradients_4/sub_2_grad/Reshape_1
s
!gradients_4/vf/Squeeze_grad/ShapeShapevf/dense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
─
#gradients_4/vf/Squeeze_grad/ReshapeReshape1gradients_4/sub_1_grad/tuple/control_dependency_1!gradients_4/vf/Squeeze_grad/Shape*
Tshape0*
T0*'
_output_shapes
:         
s
!gradients_4/vc/Squeeze_grad/ShapeShapevc/dense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
─
#gradients_4/vc/Squeeze_grad/ReshapeReshape1gradients_4/sub_2_grad/tuple/control_dependency_1!gradients_4/vc/Squeeze_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
Я
/gradients_4/vf/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_4/vf/Squeeze_grad/Reshape*
data_formatNHWC*
T0*
_output_shapes
:
Ф
4gradients_4/vf/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_4/vf/Squeeze_grad/Reshape0^gradients_4/vf/dense_2/BiasAdd_grad/BiasAddGrad
О
<gradients_4/vf/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_4/vf/Squeeze_grad/Reshape5^gradients_4/vf/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:         *
T0*6
_class,
*(loc:@gradients_4/vf/Squeeze_grad/Reshape
Ы
>gradients_4/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_4/vf/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_4/vf/dense_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*B
_class8
64loc:@gradients_4/vf/dense_2/BiasAdd_grad/BiasAddGrad
Я
/gradients_4/vc/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_4/vc/Squeeze_grad/Reshape*
_output_shapes
:*
data_formatNHWC*
T0
Ф
4gradients_4/vc/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_4/vc/Squeeze_grad/Reshape0^gradients_4/vc/dense_2/BiasAdd_grad/BiasAddGrad
О
<gradients_4/vc/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_4/vc/Squeeze_grad/Reshape5^gradients_4/vc/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:         *
T0*6
_class,
*(loc:@gradients_4/vc/Squeeze_grad/Reshape
Ы
>gradients_4/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_4/vc/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_4/vc/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*B
_class8
64loc:@gradients_4/vc/dense_2/BiasAdd_grad/BiasAddGrad
т
)gradients_4/vf/dense_2/MatMul_grad/MatMulMatMul<gradients_4/vf/dense_2/BiasAdd_grad/tuple/control_dependencyvf/dense_2/kernel/read*
transpose_a( *
T0*
transpose_b(*(
_output_shapes
:         А
╘
+gradients_4/vf/dense_2/MatMul_grad/MatMul_1MatMulvf/dense_1/Tanh<gradients_4/vf/dense_2/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	А*
T0*
transpose_a(*
transpose_b( 
Х
3gradients_4/vf/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_4/vf/dense_2/MatMul_grad/MatMul,^gradients_4/vf/dense_2/MatMul_grad/MatMul_1
Щ
;gradients_4/vf/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_4/vf/dense_2/MatMul_grad/MatMul4^gradients_4/vf/dense_2/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:         А*<
_class2
0.loc:@gradients_4/vf/dense_2/MatMul_grad/MatMul
Ц
=gradients_4/vf/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_4/vf/dense_2/MatMul_grad/MatMul_14^gradients_4/vf/dense_2/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_4/vf/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	А
т
)gradients_4/vc/dense_2/MatMul_grad/MatMulMatMul<gradients_4/vc/dense_2/BiasAdd_grad/tuple/control_dependencyvc/dense_2/kernel/read*
T0*(
_output_shapes
:         А*
transpose_b(*
transpose_a( 
╘
+gradients_4/vc/dense_2/MatMul_grad/MatMul_1MatMulvc/dense_1/Tanh<gradients_4/vc/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes
:	А*
transpose_b( 
Х
3gradients_4/vc/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_4/vc/dense_2/MatMul_grad/MatMul,^gradients_4/vc/dense_2/MatMul_grad/MatMul_1
Щ
;gradients_4/vc/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_4/vc/dense_2/MatMul_grad/MatMul4^gradients_4/vc/dense_2/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients_4/vc/dense_2/MatMul_grad/MatMul*
T0*(
_output_shapes
:         А
Ц
=gradients_4/vc/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_4/vc/dense_2/MatMul_grad/MatMul_14^gradients_4/vc/dense_2/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_4/vc/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	А
╢
)gradients_4/vf/dense_1/Tanh_grad/TanhGradTanhGradvf/dense_1/Tanh;gradients_4/vf/dense_2/MatMul_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
╢
)gradients_4/vc/dense_1/Tanh_grad/TanhGradTanhGradvc/dense_1/Tanh;gradients_4/vc/dense_2/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
ж
/gradients_4/vf/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_4/vf/dense_1/Tanh_grad/TanhGrad*
T0*
_output_shapes	
:А*
data_formatNHWC
Ъ
4gradients_4/vf/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_4/vf/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_4/vf/dense_1/Tanh_grad/TanhGrad
Ы
<gradients_4/vf/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_4/vf/dense_1/Tanh_grad/TanhGrad5^gradients_4/vf/dense_1/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_4/vf/dense_1/Tanh_grad/TanhGrad*(
_output_shapes
:         А
Ь
>gradients_4/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_4/vf/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_4/vf/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*B
_class8
64loc:@gradients_4/vf/dense_1/BiasAdd_grad/BiasAddGrad*
T0
ж
/gradients_4/vc/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_4/vc/dense_1/Tanh_grad/TanhGrad*
_output_shapes	
:А*
data_formatNHWC*
T0
Ъ
4gradients_4/vc/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_4/vc/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_4/vc/dense_1/Tanh_grad/TanhGrad
Ы
<gradients_4/vc/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_4/vc/dense_1/Tanh_grad/TanhGrad5^gradients_4/vc/dense_1/BiasAdd_grad/tuple/group_deps*
T0*(
_output_shapes
:         А*<
_class2
0.loc:@gradients_4/vc/dense_1/Tanh_grad/TanhGrad
Ь
>gradients_4/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_4/vc/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_4/vc/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*B
_class8
64loc:@gradients_4/vc/dense_1/BiasAdd_grad/BiasAddGrad
т
)gradients_4/vf/dense_1/MatMul_grad/MatMulMatMul<gradients_4/vf/dense_1/BiasAdd_grad/tuple/control_dependencyvf/dense_1/kernel/read*(
_output_shapes
:         А*
transpose_b(*
T0*
transpose_a( 
╙
+gradients_4/vf/dense_1/MatMul_grad/MatMul_1MatMulvf/dense/Tanh<gradients_4/vf/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(* 
_output_shapes
:
АА*
transpose_b( *
T0
Х
3gradients_4/vf/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_4/vf/dense_1/MatMul_grad/MatMul,^gradients_4/vf/dense_1/MatMul_grad/MatMul_1
Щ
;gradients_4/vf/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_4/vf/dense_1/MatMul_grad/MatMul4^gradients_4/vf/dense_1/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients_4/vf/dense_1/MatMul_grad/MatMul*(
_output_shapes
:         А*
T0
Ч
=gradients_4/vf/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_4/vf/dense_1/MatMul_grad/MatMul_14^gradients_4/vf/dense_1/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_4/vf/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
АА
т
)gradients_4/vc/dense_1/MatMul_grad/MatMulMatMul<gradients_4/vc/dense_1/BiasAdd_grad/tuple/control_dependencyvc/dense_1/kernel/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:         А
╙
+gradients_4/vc/dense_1/MatMul_grad/MatMul_1MatMulvc/dense/Tanh<gradients_4/vc/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
АА*
transpose_a(
Х
3gradients_4/vc/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_4/vc/dense_1/MatMul_grad/MatMul,^gradients_4/vc/dense_1/MatMul_grad/MatMul_1
Щ
;gradients_4/vc/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_4/vc/dense_1/MatMul_grad/MatMul4^gradients_4/vc/dense_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*<
_class2
0.loc:@gradients_4/vc/dense_1/MatMul_grad/MatMul
Ч
=gradients_4/vc/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_4/vc/dense_1/MatMul_grad/MatMul_14^gradients_4/vc/dense_1/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_4/vc/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
АА
▓
'gradients_4/vf/dense/Tanh_grad/TanhGradTanhGradvf/dense/Tanh;gradients_4/vf/dense_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
▓
'gradients_4/vc/dense/Tanh_grad/TanhGradTanhGradvc/dense/Tanh;gradients_4/vc/dense_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
в
-gradients_4/vf/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_4/vf/dense/Tanh_grad/TanhGrad*
_output_shapes	
:А*
data_formatNHWC*
T0
Ф
2gradients_4/vf/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_4/vf/dense/BiasAdd_grad/BiasAddGrad(^gradients_4/vf/dense/Tanh_grad/TanhGrad
У
:gradients_4/vf/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_4/vf/dense/Tanh_grad/TanhGrad3^gradients_4/vf/dense/BiasAdd_grad/tuple/group_deps*:
_class0
.,loc:@gradients_4/vf/dense/Tanh_grad/TanhGrad*
T0*(
_output_shapes
:         А
Ф
<gradients_4/vf/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_4/vf/dense/BiasAdd_grad/BiasAddGrad3^gradients_4/vf/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*@
_class6
42loc:@gradients_4/vf/dense/BiasAdd_grad/BiasAddGrad
в
-gradients_4/vc/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_4/vc/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
T0*
_output_shapes	
:А
Ф
2gradients_4/vc/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_4/vc/dense/BiasAdd_grad/BiasAddGrad(^gradients_4/vc/dense/Tanh_grad/TanhGrad
У
:gradients_4/vc/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_4/vc/dense/Tanh_grad/TanhGrad3^gradients_4/vc/dense/BiasAdd_grad/tuple/group_deps*:
_class0
.,loc:@gradients_4/vc/dense/Tanh_grad/TanhGrad*(
_output_shapes
:         А*
T0
Ф
<gradients_4/vc/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_4/vc/dense/BiasAdd_grad/BiasAddGrad3^gradients_4/vc/dense/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_4/vc/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
█
'gradients_4/vf/dense/MatMul_grad/MatMulMatMul:gradients_4/vf/dense/BiasAdd_grad/tuple/control_dependencyvf/dense/kernel/read*
transpose_a( *
transpose_b(*'
_output_shapes
:         <*
T0
╠
)gradients_4/vf/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_4/vf/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	<А*
T0*
transpose_b( 
П
1gradients_4/vf/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_4/vf/dense/MatMul_grad/MatMul*^gradients_4/vf/dense/MatMul_grad/MatMul_1
Р
9gradients_4/vf/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_4/vf/dense/MatMul_grad/MatMul2^gradients_4/vf/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:         <*:
_class0
.,loc:@gradients_4/vf/dense/MatMul_grad/MatMul*
T0
О
;gradients_4/vf/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_4/vf/dense/MatMul_grad/MatMul_12^gradients_4/vf/dense/MatMul_grad/tuple/group_deps*
_output_shapes
:	<А*<
_class2
0.loc:@gradients_4/vf/dense/MatMul_grad/MatMul_1*
T0
█
'gradients_4/vc/dense/MatMul_grad/MatMulMatMul:gradients_4/vc/dense/BiasAdd_grad/tuple/control_dependencyvc/dense/kernel/read*
T0*'
_output_shapes
:         <*
transpose_b(*
transpose_a( 
╠
)gradients_4/vc/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_4/vc/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes
:	<А*
transpose_b( 
П
1gradients_4/vc/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_4/vc/dense/MatMul_grad/MatMul*^gradients_4/vc/dense/MatMul_grad/MatMul_1
Р
9gradients_4/vc/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_4/vc/dense/MatMul_grad/MatMul2^gradients_4/vc/dense/MatMul_grad/tuple/group_deps*:
_class0
.,loc:@gradients_4/vc/dense/MatMul_grad/MatMul*
T0*'
_output_shapes
:         <
О
;gradients_4/vc/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_4/vc/dense/MatMul_grad/MatMul_12^gradients_4/vc/dense/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients_4/vc/dense/MatMul_grad/MatMul_1*
_output_shapes
:	<А*
T0
c
Reshape_42/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
Ш

Reshape_42Reshape;gradients_4/vf/dense/MatMul_grad/tuple/control_dependency_1Reshape_42/shape*
Tshape0*
T0*
_output_shapes	
:Аx
c
Reshape_43/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
Щ

Reshape_43Reshape<gradients_4/vf/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_43/shape*
Tshape0*
T0*
_output_shapes	
:А
c
Reshape_44/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
Ы

Reshape_44Reshape=gradients_4/vf/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_44/shape*
Tshape0*
T0*
_output_shapes

:АА
c
Reshape_45/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
Ы

Reshape_45Reshape>gradients_4/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_45/shape*
T0*
_output_shapes	
:А*
Tshape0
c
Reshape_46/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
Ъ

Reshape_46Reshape=gradients_4/vf/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_46/shape*
_output_shapes	
:А*
T0*
Tshape0
c
Reshape_47/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
Ъ

Reshape_47Reshape>gradients_4/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_47/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_48/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
Ш

Reshape_48Reshape;gradients_4/vc/dense/MatMul_grad/tuple/control_dependency_1Reshape_48/shape*
Tshape0*
T0*
_output_shapes	
:Аx
c
Reshape_49/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
Щ

Reshape_49Reshape<gradients_4/vc/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_49/shape*
Tshape0*
T0*
_output_shapes	
:А
c
Reshape_50/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
Ы

Reshape_50Reshape=gradients_4/vc/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_50/shape*
T0*
Tshape0*
_output_shapes

:АА
c
Reshape_51/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
Ы

Reshape_51Reshape>gradients_4/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_51/shape*
_output_shapes	
:А*
T0*
Tshape0
c
Reshape_52/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
Ъ

Reshape_52Reshape=gradients_4/vc/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_52/shape*
_output_shapes	
:А*
T0*
Tshape0
c
Reshape_53/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
Ъ

Reshape_53Reshape>gradients_4/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_53/shape*
T0*
_output_shapes
:*
Tshape0
O
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 
я
concat_5ConcatV2
Reshape_42
Reshape_43
Reshape_44
Reshape_45
Reshape_46
Reshape_47
Reshape_48
Reshape_49
Reshape_50
Reshape_51
Reshape_52
Reshape_53concat_5/axis*
T0*
N*
_output_shapes

:В№	*

Tidx0
j
PyFuncPyFuncconcat_5*
_output_shapes

:В№	*
token
pyfunc_0*
Tout
2*
Tin
2
А
Const_7Const*E
value<B:"0 <                  <                 *
dtype0*
_output_shapes
:
S
split_1/split_dimConst*
value	B : *
_output_shapes
: *
dtype0
┼
split_1SplitVPyFuncConst_7split_1/split_dim*

Tlen0*
	num_split*h
_output_shapesV
T:Аx:А:АА:А:А::Аx:А:АА:А:А:*
T0
a
Reshape_54/shapeConst*
valueB"<      *
dtype0*
_output_shapes
:
h

Reshape_54Reshapesplit_1Reshape_54/shape*
Tshape0*
_output_shapes
:	<А*
T0
[
Reshape_55/shapeConst*
_output_shapes
:*
dtype0*
valueB:А
f

Reshape_55Reshape	split_1:1Reshape_55/shape*
_output_shapes	
:А*
Tshape0*
T0
a
Reshape_56/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
k

Reshape_56Reshape	split_1:2Reshape_56/shape*
T0*
Tshape0* 
_output_shapes
:
АА
[
Reshape_57/shapeConst*
_output_shapes
:*
valueB:А*
dtype0
f

Reshape_57Reshape	split_1:3Reshape_57/shape*
_output_shapes	
:А*
T0*
Tshape0
a
Reshape_58/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
j

Reshape_58Reshape	split_1:4Reshape_58/shape*
Tshape0*
T0*
_output_shapes
:	А
Z
Reshape_59/shapeConst*
dtype0*
_output_shapes
:*
valueB:
e

Reshape_59Reshape	split_1:5Reshape_59/shape*
_output_shapes
:*
Tshape0*
T0
a
Reshape_60/shapeConst*
valueB"<      *
dtype0*
_output_shapes
:
j

Reshape_60Reshape	split_1:6Reshape_60/shape*
T0*
Tshape0*
_output_shapes
:	<А
[
Reshape_61/shapeConst*
_output_shapes
:*
dtype0*
valueB:А
f

Reshape_61Reshape	split_1:7Reshape_61/shape*
_output_shapes	
:А*
Tshape0*
T0
a
Reshape_62/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
k

Reshape_62Reshape	split_1:8Reshape_62/shape*
Tshape0* 
_output_shapes
:
АА*
T0
[
Reshape_63/shapeConst*
_output_shapes
:*
valueB:А*
dtype0
f

Reshape_63Reshape	split_1:9Reshape_63/shape*
T0*
Tshape0*
_output_shapes	
:А
a
Reshape_64/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
k

Reshape_64Reshape
split_1:10Reshape_64/shape*
Tshape0*
T0*
_output_shapes
:	А
Z
Reshape_65/shapeConst*
dtype0*
valueB:*
_output_shapes
:
f

Reshape_65Reshape
split_1:11Reshape_65/shape*
T0*
Tshape0*
_output_shapes
:
А
beta1_power/initial_valueConst*
valueB
 *fff?*
_output_shapes
: *
dtype0* 
_class
loc:@vc/dense/bias
С
beta1_power
VariableV2* 
_class
loc:@vc/dense/bias*
shape: *
_output_shapes
: *
shared_name *
	container *
dtype0
░
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: *
validate_shape(*
use_locking(
l
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias
А
beta2_power/initial_valueConst* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
dtype0*
valueB
 *w╛?
С
beta2_power
VariableV2*
shape: *
dtype0*
shared_name *
	container * 
_class
loc:@vc/dense/bias*
_output_shapes
: 
░
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias
l
beta2_power/readIdentitybeta2_power* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0
л
6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"<      *
_output_shapes
:*"
_class
loc:@vf/dense/kernel*
dtype0
Х
,vf/dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *"
_class
loc:@vf/dense/kernel*
dtype0
Ї
&vf/dense/kernel/Adam/Initializer/zerosFill6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vf/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	<А*"
_class
loc:@vf/dense/kernel*

index_type0*
T0
о
vf/dense/kernel/Adam
VariableV2*
	container *
_output_shapes
:	<А*
shared_name *"
_class
loc:@vf/dense/kernel*
dtype0*
shape:	<А
┌
vf/dense/kernel/Adam/AssignAssignvf/dense/kernel/Adam&vf/dense/kernel/Adam/Initializer/zeros*
_output_shapes
:	<А*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(
Й
vf/dense/kernel/Adam/readIdentityvf/dense/kernel/Adam*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<А
н
8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*"
_class
loc:@vf/dense/kernel*
valueB"<      
Ч
.vf/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *"
_class
loc:@vf/dense/kernel*
dtype0
·
(vf/dense/kernel/Adam_1/Initializer/zerosFill8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vf/dense/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	<А*
T0*"
_class
loc:@vf/dense/kernel*

index_type0
░
vf/dense/kernel/Adam_1
VariableV2*
shared_name *
dtype0*
	container *
shape:	<А*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<А
р
vf/dense/kernel/Adam_1/AssignAssignvf/dense/kernel/Adam_1(vf/dense/kernel/Adam_1/Initializer/zeros*
T0*
_output_shapes
:	<А*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(
Н
vf/dense/kernel/Adam_1/readIdentityvf/dense/kernel/Adam_1*
_output_shapes
:	<А*
T0*"
_class
loc:@vf/dense/kernel
Х
$vf/dense/bias/Adam/Initializer/zerosConst*
_output_shapes	
:А* 
_class
loc:@vf/dense/bias*
dtype0*
valueBА*    
в
vf/dense/bias/Adam
VariableV2* 
_class
loc:@vf/dense/bias*
_output_shapes	
:А*
shared_name *
	container *
dtype0*
shape:А
╬
vf/dense/bias/Adam/AssignAssignvf/dense/bias/Adam$vf/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_output_shapes	
:А*
validate_shape(* 
_class
loc:@vf/dense/bias

vf/dense/bias/Adam/readIdentityvf/dense/bias/Adam*
T0* 
_class
loc:@vf/dense/bias*
_output_shapes	
:А
Ч
&vf/dense/bias/Adam_1/Initializer/zerosConst*
valueBА*    *
dtype0*
_output_shapes	
:А* 
_class
loc:@vf/dense/bias
д
vf/dense/bias/Adam_1
VariableV2*
_output_shapes	
:А*
	container *
shared_name *
shape:А* 
_class
loc:@vf/dense/bias*
dtype0
╘
vf/dense/bias/Adam_1/AssignAssignvf/dense/bias/Adam_1&vf/dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
T0*
_output_shapes	
:А* 
_class
loc:@vf/dense/bias*
use_locking(
Г
vf/dense/bias/Adam_1/readIdentityvf/dense/bias/Adam_1*
T0*
_output_shapes	
:А* 
_class
loc:@vf/dense/bias
п
8vf/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"      *$
_class
loc:@vf/dense_1/kernel*
dtype0
Щ
.vf/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *$
_class
loc:@vf/dense_1/kernel
¤
(vf/dense_1/kernel/Adam/Initializer/zerosFill8vf/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vf/dense_1/kernel/Adam/Initializer/zeros/Const*

index_type0* 
_output_shapes
:
АА*$
_class
loc:@vf/dense_1/kernel*
T0
┤
vf/dense_1/kernel/Adam
VariableV2*
	container *
shape:
АА*
dtype0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
АА*
shared_name 
у
vf/dense_1/kernel/Adam/AssignAssignvf/dense_1/kernel/Adam(vf/dense_1/kernel/Adam/Initializer/zeros* 
_output_shapes
:
АА*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel
Р
vf/dense_1/kernel/Adam/readIdentityvf/dense_1/kernel/Adam*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
АА*
T0
▒
:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"      *$
_class
loc:@vf/dense_1/kernel
Ы
0vf/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *$
_class
loc:@vf/dense_1/kernel*
dtype0
Г
*vf/dense_1/kernel/Adam_1/Initializer/zerosFill:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vf/dense_1/kernel/Adam_1/Initializer/zeros/Const*

index_type0* 
_output_shapes
:
АА*
T0*$
_class
loc:@vf/dense_1/kernel
╢
vf/dense_1/kernel/Adam_1
VariableV2*$
_class
loc:@vf/dense_1/kernel*
dtype0*
	container * 
_output_shapes
:
АА*
shape:
АА*
shared_name 
щ
vf/dense_1/kernel/Adam_1/AssignAssignvf/dense_1/kernel/Adam_1*vf/dense_1/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
АА*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
T0
Ф
vf/dense_1/kernel/Adam_1/readIdentityvf/dense_1/kernel/Adam_1* 
_output_shapes
:
АА*$
_class
loc:@vf/dense_1/kernel*
T0
Щ
&vf/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@vf/dense_1/bias*
dtype0*
valueBА*    *
_output_shapes	
:А
ж
vf/dense_1/bias/Adam
VariableV2*
shape:А*"
_class
loc:@vf/dense_1/bias*
dtype0*
	container *
shared_name *
_output_shapes	
:А
╓
vf/dense_1/bias/Adam/AssignAssignvf/dense_1/bias/Adam&vf/dense_1/bias/Adam/Initializer/zeros*
T0*
_output_shapes	
:А*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
use_locking(
Е
vf/dense_1/bias/Adam/readIdentityvf/dense_1/bias/Adam*
_output_shapes	
:А*"
_class
loc:@vf/dense_1/bias*
T0
Ы
(vf/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*"
_class
loc:@vf/dense_1/bias*
valueBА*    *
_output_shapes	
:А
и
vf/dense_1/bias/Adam_1
VariableV2*
dtype0*
	container *
shape:А*
shared_name *
_output_shapes	
:А*"
_class
loc:@vf/dense_1/bias
▄
vf/dense_1/bias/Adam_1/AssignAssignvf/dense_1/bias/Adam_1(vf/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:А
Й
vf/dense_1/bias/Adam_1/readIdentityvf/dense_1/bias/Adam_1*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:А
е
(vf/dense_2/kernel/Adam/Initializer/zerosConst*
valueB	А*    *
dtype0*
_output_shapes
:	А*$
_class
loc:@vf/dense_2/kernel
▓
vf/dense_2/kernel/Adam
VariableV2*
_output_shapes
:	А*
shape:	А*$
_class
loc:@vf/dense_2/kernel*
	container *
shared_name *
dtype0
т
vf/dense_2/kernel/Adam/AssignAssignvf/dense_2/kernel/Adam(vf/dense_2/kernel/Adam/Initializer/zeros*
_output_shapes
:	А*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
П
vf/dense_2/kernel/Adam/readIdentityvf/dense_2/kernel/Adam*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	А*
T0
з
*vf/dense_2/kernel/Adam_1/Initializer/zerosConst*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	А*
dtype0*
valueB	А*    
┤
vf/dense_2/kernel/Adam_1
VariableV2*
_output_shapes
:	А*$
_class
loc:@vf/dense_2/kernel*
shared_name *
	container *
shape:	А*
dtype0
ш
vf/dense_2/kernel/Adam_1/AssignAssignvf/dense_2/kernel/Adam_1*vf/dense_2/kernel/Adam_1/Initializer/zeros*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	А*
T0
У
vf/dense_2/kernel/Adam_1/readIdentityvf/dense_2/kernel/Adam_1*
_output_shapes
:	А*$
_class
loc:@vf/dense_2/kernel*
T0
Ч
&vf/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
д
vf/dense_2/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
shared_name *"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
╒
vf/dense_2/bias/Adam/AssignAssignvf/dense_2/bias/Adam&vf/dense_2/bias/Adam/Initializer/zeros*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:
Д
vf/dense_2/bias/Adam/readIdentityvf/dense_2/bias/Adam*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias
Щ
(vf/dense_2/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@vf/dense_2/bias*
valueB*    *
_output_shapes
:*
dtype0
ж
vf/dense_2/bias/Adam_1
VariableV2*
_output_shapes
:*
shared_name *
shape:*
dtype0*
	container *"
_class
loc:@vf/dense_2/bias
█
vf/dense_2/bias/Adam_1/AssignAssignvf/dense_2/bias/Adam_1(vf/dense_2/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0
И
vf/dense_2/bias/Adam_1/readIdentityvf/dense_2/bias/Adam_1*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0
л
6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@vc/dense/kernel*
_output_shapes
:*
dtype0*
valueB"<      
Х
,vc/dense/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*"
_class
loc:@vc/dense/kernel*
valueB
 *    *
_output_shapes
: 
Ї
&vc/dense/kernel/Adam/Initializer/zerosFill6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vc/dense/kernel/Adam/Initializer/zeros/Const*"
_class
loc:@vc/dense/kernel*

index_type0*
_output_shapes
:	<А*
T0
о
vc/dense/kernel/Adam
VariableV2*"
_class
loc:@vc/dense/kernel*
	container *
dtype0*
shape:	<А*
shared_name *
_output_shapes
:	<А
┌
vc/dense/kernel/Adam/AssignAssignvc/dense/kernel/Adam&vc/dense/kernel/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<А
Й
vc/dense/kernel/Adam/readIdentityvc/dense/kernel/Adam*
T0*
_output_shapes
:	<А*"
_class
loc:@vc/dense/kernel
н
8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*"
_class
loc:@vc/dense/kernel*
valueB"<      *
dtype0
Ч
.vc/dense/kernel/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@vc/dense/kernel*
dtype0*
valueB
 *    *
_output_shapes
: 
·
(vc/dense/kernel/Adam_1/Initializer/zerosFill8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vc/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*"
_class
loc:@vc/dense/kernel*

index_type0*
_output_shapes
:	<А
░
vc/dense/kernel/Adam_1
VariableV2*
shape:	<А*
	container *
dtype0*
shared_name *"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<А
р
vc/dense/kernel/Adam_1/AssignAssignvc/dense/kernel/Adam_1(vc/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	<А*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vc/dense/kernel
Н
vc/dense/kernel/Adam_1/readIdentityvc/dense/kernel/Adam_1*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<А
Х
$vc/dense/bias/Adam/Initializer/zerosConst*
dtype0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:А*
valueBА*    
в
vc/dense/bias/Adam
VariableV2*
_output_shapes	
:А*
shared_name *
	container * 
_class
loc:@vc/dense/bias*
dtype0*
shape:А
╬
vc/dense/bias/Adam/AssignAssignvc/dense/bias/Adam$vc/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:А* 
_class
loc:@vc/dense/bias

vc/dense/bias/Adam/readIdentityvc/dense/bias/Adam* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:А
Ч
&vc/dense/bias/Adam_1/Initializer/zerosConst*
dtype0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:А*
valueBА*    
д
vc/dense/bias/Adam_1
VariableV2* 
_class
loc:@vc/dense/bias*
dtype0*
_output_shapes	
:А*
shared_name *
shape:А*
	container 
╘
vc/dense/bias/Adam_1/AssignAssignvc/dense/bias/Adam_1&vc/dense/bias/Adam_1/Initializer/zeros* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
Г
vc/dense/bias/Adam_1/readIdentityvc/dense/bias/Adam_1* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:А
п
8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@vc/dense_1/kernel*
valueB"      *
_output_shapes
:*
dtype0
Щ
.vc/dense_1/kernel/Adam/Initializer/zeros/ConstConst*$
_class
loc:@vc/dense_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
¤
(vc/dense_1/kernel/Adam/Initializer/zerosFill8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vc/dense_1/kernel/Adam/Initializer/zeros/Const*$
_class
loc:@vc/dense_1/kernel*
T0*

index_type0* 
_output_shapes
:
АА
┤
vc/dense_1/kernel/Adam
VariableV2* 
_output_shapes
:
АА*
shape:
АА*$
_class
loc:@vc/dense_1/kernel*
shared_name *
	container *
dtype0
у
vc/dense_1/kernel/Adam/AssignAssignvc/dense_1/kernel/Adam(vc/dense_1/kernel/Adam/Initializer/zeros*
validate_shape(*
T0* 
_output_shapes
:
АА*$
_class
loc:@vc/dense_1/kernel*
use_locking(
Р
vc/dense_1/kernel/Adam/readIdentityvc/dense_1/kernel/Adam*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:
АА
▒
:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@vc/dense_1/kernel*
dtype0*
valueB"      *
_output_shapes
:
Ы
0vc/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@vc/dense_1/kernel*
_output_shapes
: *
dtype0
Г
*vc/dense_1/kernel/Adam_1/Initializer/zerosFill:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vc/dense_1/kernel/Adam_1/Initializer/zeros/Const*

index_type0*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
АА
╢
vc/dense_1/kernel/Adam_1
VariableV2* 
_output_shapes
:
АА*$
_class
loc:@vc/dense_1/kernel*
shape:
АА*
dtype0*
	container *
shared_name 
щ
vc/dense_1/kernel/Adam_1/AssignAssignvc/dense_1/kernel/Adam_1*vc/dense_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
АА
Ф
vc/dense_1/kernel/Adam_1/readIdentityvc/dense_1/kernel/Adam_1*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:
АА
Щ
&vc/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:А*
valueBА*    *
dtype0
ж
vc/dense_1/bias/Adam
VariableV2*"
_class
loc:@vc/dense_1/bias*
	container *
dtype0*
shared_name *
_output_shapes	
:А*
shape:А
╓
vc/dense_1/bias/Adam/AssignAssignvc/dense_1/bias/Adam&vc/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:А*
T0*"
_class
loc:@vc/dense_1/bias
Е
vc/dense_1/bias/Adam/readIdentityvc/dense_1/bias/Adam*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:А*
T0
Ы
(vc/dense_1/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:А*
valueBА*    *"
_class
loc:@vc/dense_1/bias*
dtype0
и
vc/dense_1/bias/Adam_1
VariableV2*
dtype0*
	container *
_output_shapes	
:А*
shared_name *
shape:А*"
_class
loc:@vc/dense_1/bias
▄
vc/dense_1/bias/Adam_1/AssignAssignvc/dense_1/bias/Adam_1(vc/dense_1/bias/Adam_1/Initializer/zeros*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:А*
use_locking(
Й
vc/dense_1/bias/Adam_1/readIdentityvc/dense_1/bias/Adam_1*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:А
е
(vc/dense_2/kernel/Adam/Initializer/zerosConst*
dtype0*$
_class
loc:@vc/dense_2/kernel*
valueB	А*    *
_output_shapes
:	А
▓
vc/dense_2/kernel/Adam
VariableV2*
dtype0*
	container *$
_class
loc:@vc/dense_2/kernel*
shared_name *
shape:	А*
_output_shapes
:	А
т
vc/dense_2/kernel/Adam/AssignAssignvc/dense_2/kernel/Adam(vc/dense_2/kernel/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel
П
vc/dense_2/kernel/Adam/readIdentityvc/dense_2/kernel/Adam*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel*
T0
з
*vc/dense_2/kernel/Adam_1/Initializer/zerosConst*
valueB	А*    *
_output_shapes
:	А*
dtype0*$
_class
loc:@vc/dense_2/kernel
┤
vc/dense_2/kernel/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
shape:	А*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel
ш
vc/dense_2/kernel/Adam_1/AssignAssignvc/dense_2/kernel/Adam_1*vc/dense_2/kernel/Adam_1/Initializer/zeros*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	А*
validate_shape(
У
vc/dense_2/kernel/Adam_1/readIdentityvc/dense_2/kernel/Adam_1*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	А
Ч
&vc/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *"
_class
loc:@vc/dense_2/bias
д
vc/dense_2/bias/Adam
VariableV2*
dtype0*
shared_name *
	container *"
_class
loc:@vc/dense_2/bias*
shape:*
_output_shapes
:
╒
vc/dense_2/bias/Adam/AssignAssignvc/dense_2/bias/Adam&vc/dense_2/bias/Adam/Initializer/zeros*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Д
vc/dense_2/bias/Adam/readIdentityvc/dense_2/bias/Adam*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
Щ
(vc/dense_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*
dtype0*"
_class
loc:@vc/dense_2/bias*
valueB*    
ж
vc/dense_2/bias/Adam_1
VariableV2*"
_class
loc:@vc/dense_2/bias*
dtype0*
_output_shapes
:*
shared_name *
shape:*
	container 
█
vc/dense_2/bias/Adam_1/AssignAssignvc/dense_2/bias/Adam_1(vc/dense_2/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias*
use_locking(
И
vc/dense_2/bias/Adam_1/readIdentityvc/dense_2/bias/Adam_1*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *oГ:
O

Adam/beta1Const*
valueB
 *fff?*
_output_shapes
: *
dtype0
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w╛?
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *w╠+2*
dtype0
╨
%Adam/update_vf/dense/kernel/ApplyAdam	ApplyAdamvf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_54*
use_locking( *"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	<А*
use_nesterov( 
┬
#Adam/update_vf/dense/bias/ApplyAdam	ApplyAdamvf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_55*
use_locking( * 
_class
loc:@vf/dense/bias*
use_nesterov( *
_output_shapes	
:А*
T0
█
'Adam/update_vf/dense_1/kernel/ApplyAdam	ApplyAdamvf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_56*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
АА*
T0*
use_locking( *
use_nesterov( 
╠
%Adam/update_vf/dense_1/bias/ApplyAdam	ApplyAdamvf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_57*
use_nesterov( *"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:А*
use_locking( *
T0
┌
'Adam/update_vf/dense_2/kernel/ApplyAdam	ApplyAdamvf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_58*
use_nesterov( *
T0*
use_locking( *$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	А
╦
%Adam/update_vf/dense_2/bias/ApplyAdam	ApplyAdamvf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_59*"
_class
loc:@vf/dense_2/bias*
use_nesterov( *
T0*
_output_shapes
:*
use_locking( 
╨
%Adam/update_vc/dense/kernel/ApplyAdam	ApplyAdamvc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_60*
use_nesterov( *
T0*
_output_shapes
:	<А*"
_class
loc:@vc/dense/kernel*
use_locking( 
┬
#Adam/update_vc/dense/bias/ApplyAdam	ApplyAdamvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_61*
use_nesterov( *
_output_shapes	
:А* 
_class
loc:@vc/dense/bias*
use_locking( *
T0
█
'Adam/update_vc/dense_1/kernel/ApplyAdam	ApplyAdamvc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_62* 
_output_shapes
:
АА*
T0*
use_locking( *$
_class
loc:@vc/dense_1/kernel*
use_nesterov( 
╠
%Adam/update_vc/dense_1/bias/ApplyAdam	ApplyAdamvc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_63*
use_locking( *
use_nesterov( *
T0*
_output_shapes	
:А*"
_class
loc:@vc/dense_1/bias
┌
'Adam/update_vc/dense_2/kernel/ApplyAdam	ApplyAdamvc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_64*
T0*
use_nesterov( *
use_locking( *$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	А
╦
%Adam/update_vc/dense_2/bias/ApplyAdam	ApplyAdamvc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_65*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking( *
use_nesterov( *
T0
╘
Adam/mulMulbeta1_power/read
Adam/beta1$^Adam/update_vc/dense/bias/ApplyAdam&^Adam/update_vc/dense/kernel/ApplyAdam&^Adam/update_vc/dense_1/bias/ApplyAdam(^Adam/update_vc/dense_1/kernel/ApplyAdam&^Adam/update_vc/dense_2/bias/ApplyAdam(^Adam/update_vc/dense_2/kernel/ApplyAdam$^Adam/update_vf/dense/bias/ApplyAdam&^Adam/update_vf/dense/kernel/ApplyAdam&^Adam/update_vf/dense_1/bias/ApplyAdam(^Adam/update_vf/dense_1/kernel/ApplyAdam&^Adam/update_vf/dense_2/bias/ApplyAdam(^Adam/update_vf/dense_2/kernel/ApplyAdam*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0
Ш
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: 
╓

Adam/mul_1Mulbeta2_power/read
Adam/beta2$^Adam/update_vc/dense/bias/ApplyAdam&^Adam/update_vc/dense/kernel/ApplyAdam&^Adam/update_vc/dense_1/bias/ApplyAdam(^Adam/update_vc/dense_1/kernel/ApplyAdam&^Adam/update_vc/dense_2/bias/ApplyAdam(^Adam/update_vc/dense_2/kernel/ApplyAdam$^Adam/update_vf/dense/bias/ApplyAdam&^Adam/update_vf/dense/kernel/ApplyAdam&^Adam/update_vf/dense_1/bias/ApplyAdam(^Adam/update_vf/dense_1/kernel/ApplyAdam&^Adam/update_vf/dense_2/bias/ApplyAdam(^Adam/update_vf/dense_2/kernel/ApplyAdam* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: 
Ь
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
use_locking( *
T0
О
AdamNoOp^Adam/Assign^Adam/Assign_1$^Adam/update_vc/dense/bias/ApplyAdam&^Adam/update_vc/dense/kernel/ApplyAdam&^Adam/update_vc/dense_1/bias/ApplyAdam(^Adam/update_vc/dense_1/kernel/ApplyAdam&^Adam/update_vc/dense_2/bias/ApplyAdam(^Adam/update_vc/dense_2/kernel/ApplyAdam$^Adam/update_vf/dense/bias/ApplyAdam&^Adam/update_vf/dense/kernel/ApplyAdam&^Adam/update_vf/dense_1/bias/ApplyAdam(^Adam/update_vf/dense_1/kernel/ApplyAdam&^Adam/update_vf/dense_2/bias/ApplyAdam(^Adam/update_vf/dense_2/kernel/ApplyAdam
j
Reshape_66/shapeConst^Adam*
valueB:
         *
dtype0*
_output_shapes
:
q

Reshape_66Reshapevf/dense/kernel/readReshape_66/shape*
Tshape0*
T0*
_output_shapes	
:Аx
j
Reshape_67/shapeConst^Adam*
dtype0*
valueB:
         *
_output_shapes
:
o

Reshape_67Reshapevf/dense/bias/readReshape_67/shape*
_output_shapes	
:А*
Tshape0*
T0
j
Reshape_68/shapeConst^Adam*
dtype0*
valueB:
         *
_output_shapes
:
t

Reshape_68Reshapevf/dense_1/kernel/readReshape_68/shape*
Tshape0*
_output_shapes

:АА*
T0
j
Reshape_69/shapeConst^Adam*
dtype0*
valueB:
         *
_output_shapes
:
q

Reshape_69Reshapevf/dense_1/bias/readReshape_69/shape*
_output_shapes	
:А*
T0*
Tshape0
j
Reshape_70/shapeConst^Adam*
valueB:
         *
_output_shapes
:*
dtype0
s

Reshape_70Reshapevf/dense_2/kernel/readReshape_70/shape*
Tshape0*
_output_shapes	
:А*
T0
j
Reshape_71/shapeConst^Adam*
_output_shapes
:*
valueB:
         *
dtype0
p

Reshape_71Reshapevf/dense_2/bias/readReshape_71/shape*
_output_shapes
:*
T0*
Tshape0
j
Reshape_72/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:
         
q

Reshape_72Reshapevc/dense/kernel/readReshape_72/shape*
T0*
Tshape0*
_output_shapes	
:Аx
j
Reshape_73/shapeConst^Adam*
valueB:
         *
dtype0*
_output_shapes
:
o

Reshape_73Reshapevc/dense/bias/readReshape_73/shape*
T0*
Tshape0*
_output_shapes	
:А
j
Reshape_74/shapeConst^Adam*
valueB:
         *
_output_shapes
:*
dtype0
t

Reshape_74Reshapevc/dense_1/kernel/readReshape_74/shape*
_output_shapes

:АА*
Tshape0*
T0
j
Reshape_75/shapeConst^Adam*
dtype0*
valueB:
         *
_output_shapes
:
q

Reshape_75Reshapevc/dense_1/bias/readReshape_75/shape*
Tshape0*
_output_shapes	
:А*
T0
j
Reshape_76/shapeConst^Adam*
valueB:
         *
_output_shapes
:*
dtype0
s

Reshape_76Reshapevc/dense_2/kernel/readReshape_76/shape*
T0*
Tshape0*
_output_shapes	
:А
j
Reshape_77/shapeConst^Adam*
valueB:
         *
_output_shapes
:*
dtype0
p

Reshape_77Reshapevc/dense_2/bias/readReshape_77/shape*
_output_shapes
:*
T0*
Tshape0
V
concat_6/axisConst^Adam*
dtype0*
value	B : *
_output_shapes
: 
я
concat_6ConcatV2
Reshape_66
Reshape_67
Reshape_68
Reshape_69
Reshape_70
Reshape_71
Reshape_72
Reshape_73
Reshape_74
Reshape_75
Reshape_76
Reshape_77concat_6/axis*
T0*
N*
_output_shapes

:В№	*

Tidx0
h
PyFunc_1PyFuncconcat_6*
Tout
2*
Tin
2*
token
pyfunc_1*
_output_shapes
:
З
Const_8Const^Adam*
_output_shapes
:*
dtype0*E
value<B:"0 <                  <                 
Z
split_2/split_dimConst^Adam*
_output_shapes
: *
value	B : *
dtype0
г
split_2SplitVPyFunc_1Const_8split_2/split_dim*D
_output_shapes2
0::::::::::::*

Tlen0*
T0*
	num_split
h
Reshape_78/shapeConst^Adam*
dtype0*
valueB"<      *
_output_shapes
:
h

Reshape_78Reshapesplit_2Reshape_78/shape*
T0*
Tshape0*
_output_shapes
:	<А
b
Reshape_79/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:А
f

Reshape_79Reshape	split_2:1Reshape_79/shape*
_output_shapes	
:А*
Tshape0*
T0
h
Reshape_80/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB"      
k

Reshape_80Reshape	split_2:2Reshape_80/shape* 
_output_shapes
:
АА*
Tshape0*
T0
b
Reshape_81/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:А
f

Reshape_81Reshape	split_2:3Reshape_81/shape*
T0*
Tshape0*
_output_shapes	
:А
h
Reshape_82/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB"      
j

Reshape_82Reshape	split_2:4Reshape_82/shape*
Tshape0*
T0*
_output_shapes
:	А
a
Reshape_83/shapeConst^Adam*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_83Reshape	split_2:5Reshape_83/shape*
Tshape0*
T0*
_output_shapes
:
h
Reshape_84/shapeConst^Adam*
valueB"<      *
dtype0*
_output_shapes
:
j

Reshape_84Reshape	split_2:6Reshape_84/shape*
_output_shapes
:	<А*
T0*
Tshape0
b
Reshape_85/shapeConst^Adam*
_output_shapes
:*
valueB:А*
dtype0
f

Reshape_85Reshape	split_2:7Reshape_85/shape*
Tshape0*
T0*
_output_shapes	
:А
h
Reshape_86/shapeConst^Adam*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_86Reshape	split_2:8Reshape_86/shape* 
_output_shapes
:
АА*
T0*
Tshape0
b
Reshape_87/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:А
f

Reshape_87Reshape	split_2:9Reshape_87/shape*
T0*
Tshape0*
_output_shapes	
:А
h
Reshape_88/shapeConst^Adam*
dtype0*
valueB"      *
_output_shapes
:
k

Reshape_88Reshape
split_2:10Reshape_88/shape*
Tshape0*
_output_shapes
:	А*
T0
a
Reshape_89/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:
f

Reshape_89Reshape
split_2:11Reshape_89/shape*
Tshape0*
_output_shapes
:*
T0
ж
Assign_7Assignvf/dense/kernel
Reshape_78*
T0*
validate_shape(*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	<А
Ю
Assign_8Assignvf/dense/bias
Reshape_79*
_output_shapes	
:А*
use_locking(* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0
л
Assign_9Assignvf/dense_1/kernel
Reshape_80*
T0*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:
АА*
validate_shape(
г
	Assign_10Assignvf/dense_1/bias
Reshape_81*
_output_shapes	
:А*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(
л
	Assign_11Assignvf/dense_2/kernel
Reshape_82*
_output_shapes
:	А*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
T0
в
	Assign_12Assignvf/dense_2/bias
Reshape_83*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*"
_class
loc:@vf/dense_2/bias
з
	Assign_13Assignvc/dense/kernel
Reshape_84*
validate_shape(*
T0*
_output_shapes
:	<А*"
_class
loc:@vc/dense/kernel*
use_locking(
Я
	Assign_14Assignvc/dense/bias
Reshape_85*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:А* 
_class
loc:@vc/dense/bias
м
	Assign_15Assignvc/dense_1/kernel
Reshape_86*
use_locking(* 
_output_shapes
:
АА*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
г
	Assign_16Assignvc/dense_1/bias
Reshape_87*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:А*
use_locking(
л
	Assign_17Assignvc/dense_2/kernel
Reshape_88*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	А*
T0
в
	Assign_18Assignvc/dense_2/bias
Reshape_89*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
и
group_deps_1NoOp^Adam
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18	^Assign_7	^Assign_8	^Assign_9
*
group_deps_2NoOp^Adam^group_deps_1
В

initNoOp^beta1_power/Assign^beta2_power/Assign^pi/dense/bias/Assign^pi/dense/kernel/Assign^pi/dense_1/bias/Assign^pi/dense_1/kernel/Assign^pi/dense_2/bias/Assign^pi/dense_2/kernel/Assign^pi/log_std/Assign^vc/dense/bias/Adam/Assign^vc/dense/bias/Adam_1/Assign^vc/dense/bias/Assign^vc/dense/kernel/Adam/Assign^vc/dense/kernel/Adam_1/Assign^vc/dense/kernel/Assign^vc/dense_1/bias/Adam/Assign^vc/dense_1/bias/Adam_1/Assign^vc/dense_1/bias/Assign^vc/dense_1/kernel/Adam/Assign ^vc/dense_1/kernel/Adam_1/Assign^vc/dense_1/kernel/Assign^vc/dense_2/bias/Adam/Assign^vc/dense_2/bias/Adam_1/Assign^vc/dense_2/bias/Assign^vc/dense_2/kernel/Adam/Assign ^vc/dense_2/kernel/Adam_1/Assign^vc/dense_2/kernel/Assign^vf/dense/bias/Adam/Assign^vf/dense/bias/Adam_1/Assign^vf/dense/bias/Assign^vf/dense/kernel/Adam/Assign^vf/dense/kernel/Adam_1/Assign^vf/dense/kernel/Assign^vf/dense_1/bias/Adam/Assign^vf/dense_1/bias/Adam_1/Assign^vf/dense_1/bias/Assign^vf/dense_1/kernel/Adam/Assign ^vf/dense_1/kernel/Adam_1/Assign^vf/dense_1/kernel/Assign^vf/dense_2/bias/Adam/Assign^vf/dense_2/bias/Adam_1/Assign^vf/dense_2/bias/Assign^vf/dense_2/kernel/Adam/Assign ^vf/dense_2/kernel/Adam_1/Assign^vf/dense_2/kernel/Assign
c
Reshape_90/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
q

Reshape_90Reshapepi/dense/kernel/readReshape_90/shape*
T0*
Tshape0*
_output_shapes	
:Аx
c
Reshape_91/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
o

Reshape_91Reshapepi/dense/bias/readReshape_91/shape*
T0*
Tshape0*
_output_shapes	
:А
c
Reshape_92/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
t

Reshape_92Reshapepi/dense_1/kernel/readReshape_92/shape*
T0*
Tshape0*
_output_shapes

:АА
c
Reshape_93/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
q

Reshape_93Reshapepi/dense_1/bias/readReshape_93/shape*
Tshape0*
T0*
_output_shapes	
:А
c
Reshape_94/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
s

Reshape_94Reshapepi/dense_2/kernel/readReshape_94/shape*
_output_shapes	
:А*
T0*
Tshape0
c
Reshape_95/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
p

Reshape_95Reshapepi/dense_2/bias/readReshape_95/shape*
_output_shapes
:*
Tshape0*
T0
c
Reshape_96/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
k

Reshape_96Reshapepi/log_std/readReshape_96/shape*
Tshape0*
T0*
_output_shapes
:
c
Reshape_97/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
q

Reshape_97Reshapevf/dense/kernel/readReshape_97/shape*
T0*
Tshape0*
_output_shapes	
:Аx
c
Reshape_98/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
o

Reshape_98Reshapevf/dense/bias/readReshape_98/shape*
_output_shapes	
:А*
Tshape0*
T0
c
Reshape_99/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
t

Reshape_99Reshapevf/dense_1/kernel/readReshape_99/shape*
_output_shapes

:АА*
T0*
Tshape0
d
Reshape_100/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
s
Reshape_100Reshapevf/dense_1/bias/readReshape_100/shape*
Tshape0*
T0*
_output_shapes	
:А
d
Reshape_101/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
u
Reshape_101Reshapevf/dense_2/kernel/readReshape_101/shape*
T0*
_output_shapes	
:А*
Tshape0
d
Reshape_102/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
r
Reshape_102Reshapevf/dense_2/bias/readReshape_102/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_103/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
s
Reshape_103Reshapevc/dense/kernel/readReshape_103/shape*
_output_shapes	
:Аx*
Tshape0*
T0
d
Reshape_104/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
q
Reshape_104Reshapevc/dense/bias/readReshape_104/shape*
T0*
_output_shapes	
:А*
Tshape0
d
Reshape_105/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
v
Reshape_105Reshapevc/dense_1/kernel/readReshape_105/shape*
T0*
Tshape0*
_output_shapes

:АА
d
Reshape_106/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
s
Reshape_106Reshapevc/dense_1/bias/readReshape_106/shape*
_output_shapes	
:А*
T0*
Tshape0
d
Reshape_107/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
u
Reshape_107Reshapevc/dense_2/kernel/readReshape_107/shape*
Tshape0*
T0*
_output_shapes	
:А
d
Reshape_108/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
r
Reshape_108Reshapevc/dense_2/bias/readReshape_108/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_109/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
n
Reshape_109Reshapebeta1_power/readReshape_109/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_110/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
n
Reshape_110Reshapebeta2_power/readReshape_110/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_111/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
x
Reshape_111Reshapevf/dense/kernel/Adam/readReshape_111/shape*
T0*
Tshape0*
_output_shapes	
:Аx
d
Reshape_112/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
z
Reshape_112Reshapevf/dense/kernel/Adam_1/readReshape_112/shape*
T0*
_output_shapes	
:Аx*
Tshape0
d
Reshape_113/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
v
Reshape_113Reshapevf/dense/bias/Adam/readReshape_113/shape*
_output_shapes	
:А*
T0*
Tshape0
d
Reshape_114/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
x
Reshape_114Reshapevf/dense/bias/Adam_1/readReshape_114/shape*
Tshape0*
_output_shapes	
:А*
T0
d
Reshape_115/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
{
Reshape_115Reshapevf/dense_1/kernel/Adam/readReshape_115/shape*
T0*
Tshape0*
_output_shapes

:АА
d
Reshape_116/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
}
Reshape_116Reshapevf/dense_1/kernel/Adam_1/readReshape_116/shape*
Tshape0*
T0*
_output_shapes

:АА
d
Reshape_117/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
x
Reshape_117Reshapevf/dense_1/bias/Adam/readReshape_117/shape*
_output_shapes	
:А*
Tshape0*
T0
d
Reshape_118/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
z
Reshape_118Reshapevf/dense_1/bias/Adam_1/readReshape_118/shape*
_output_shapes	
:А*
Tshape0*
T0
d
Reshape_119/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
z
Reshape_119Reshapevf/dense_2/kernel/Adam/readReshape_119/shape*
Tshape0*
T0*
_output_shapes	
:А
d
Reshape_120/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
|
Reshape_120Reshapevf/dense_2/kernel/Adam_1/readReshape_120/shape*
Tshape0*
_output_shapes	
:А*
T0
d
Reshape_121/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
w
Reshape_121Reshapevf/dense_2/bias/Adam/readReshape_121/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_122/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
y
Reshape_122Reshapevf/dense_2/bias/Adam_1/readReshape_122/shape*
T0*
_output_shapes
:*
Tshape0
d
Reshape_123/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
x
Reshape_123Reshapevc/dense/kernel/Adam/readReshape_123/shape*
Tshape0*
T0*
_output_shapes	
:Аx
d
Reshape_124/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
z
Reshape_124Reshapevc/dense/kernel/Adam_1/readReshape_124/shape*
_output_shapes	
:Аx*
Tshape0*
T0
d
Reshape_125/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
v
Reshape_125Reshapevc/dense/bias/Adam/readReshape_125/shape*
Tshape0*
T0*
_output_shapes	
:А
d
Reshape_126/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
x
Reshape_126Reshapevc/dense/bias/Adam_1/readReshape_126/shape*
T0*
Tshape0*
_output_shapes	
:А
d
Reshape_127/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
{
Reshape_127Reshapevc/dense_1/kernel/Adam/readReshape_127/shape*
_output_shapes

:АА*
T0*
Tshape0
d
Reshape_128/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
}
Reshape_128Reshapevc/dense_1/kernel/Adam_1/readReshape_128/shape*
Tshape0*
_output_shapes

:АА*
T0
d
Reshape_129/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
x
Reshape_129Reshapevc/dense_1/bias/Adam/readReshape_129/shape*
_output_shapes	
:А*
T0*
Tshape0
d
Reshape_130/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
z
Reshape_130Reshapevc/dense_1/bias/Adam_1/readReshape_130/shape*
T0*
_output_shapes	
:А*
Tshape0
d
Reshape_131/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
z
Reshape_131Reshapevc/dense_2/kernel/Adam/readReshape_131/shape*
T0*
Tshape0*
_output_shapes	
:А
d
Reshape_132/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
|
Reshape_132Reshapevc/dense_2/kernel/Adam_1/readReshape_132/shape*
Tshape0*
T0*
_output_shapes	
:А
d
Reshape_133/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
w
Reshape_133Reshapevc/dense_2/bias/Adam/readReshape_133/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_134/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
y
Reshape_134Reshapevc/dense_2/bias/Adam_1/readReshape_134/shape*
_output_shapes
:*
Tshape0*
T0
O
concat_7/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ю
concat_7ConcatV2
Reshape_90
Reshape_91
Reshape_92
Reshape_93
Reshape_94
Reshape_95
Reshape_96
Reshape_97
Reshape_98
Reshape_99Reshape_100Reshape_101Reshape_102Reshape_103Reshape_104Reshape_105Reshape_106Reshape_107Reshape_108Reshape_109Reshape_110Reshape_111Reshape_112Reshape_113Reshape_114Reshape_115Reshape_116Reshape_117Reshape_118Reshape_119Reshape_120Reshape_121Reshape_122Reshape_123Reshape_124Reshape_125Reshape_126Reshape_127Reshape_128Reshape_129Reshape_130Reshape_131Reshape_132Reshape_133Reshape_134concat_7/axis*
T0*

Tidx0*
N-*
_output_shapes

:МЇ"
h
PyFunc_2PyFuncconcat_7*
_output_shapes
:*
Tout
2*
Tin
2*
token
pyfunc_2
И
Const_9Const*
dtype0*
_output_shapes
:-*╠
value┬B┐-"┤ <                     <                  <                        <   <                                 <   <                                
S
split_3/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
к
split_3SplitVPyFunc_2Const_9split_3/split_dim*
	num_split-*

Tlen0*╩
_output_shapes╖
┤:::::::::::::::::::::::::::::::::::::::::::::*
T0
b
Reshape_135/shapeConst*
valueB"<      *
dtype0*
_output_shapes
:
j
Reshape_135Reshapesplit_3Reshape_135/shape*
Tshape0*
_output_shapes
:	<А*
T0
\
Reshape_136/shapeConst*
dtype0*
_output_shapes
:*
valueB:А
h
Reshape_136Reshape	split_3:1Reshape_136/shape*
Tshape0*
T0*
_output_shapes	
:А
b
Reshape_137/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
m
Reshape_137Reshape	split_3:2Reshape_137/shape* 
_output_shapes
:
АА*
T0*
Tshape0
\
Reshape_138/shapeConst*
_output_shapes
:*
dtype0*
valueB:А
h
Reshape_138Reshape	split_3:3Reshape_138/shape*
T0*
Tshape0*
_output_shapes	
:А
b
Reshape_139/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
l
Reshape_139Reshape	split_3:4Reshape_139/shape*
_output_shapes
:	А*
Tshape0*
T0
[
Reshape_140/shapeConst*
valueB:*
_output_shapes
:*
dtype0
g
Reshape_140Reshape	split_3:5Reshape_140/shape*
_output_shapes
:*
T0*
Tshape0
[
Reshape_141/shapeConst*
dtype0*
_output_shapes
:*
valueB:
g
Reshape_141Reshape	split_3:6Reshape_141/shape*
_output_shapes
:*
T0*
Tshape0
b
Reshape_142/shapeConst*
dtype0*
valueB"<      *
_output_shapes
:
l
Reshape_142Reshape	split_3:7Reshape_142/shape*
_output_shapes
:	<А*
T0*
Tshape0
\
Reshape_143/shapeConst*
dtype0*
valueB:А*
_output_shapes
:
h
Reshape_143Reshape	split_3:8Reshape_143/shape*
_output_shapes	
:А*
T0*
Tshape0
b
Reshape_144/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
m
Reshape_144Reshape	split_3:9Reshape_144/shape*
T0* 
_output_shapes
:
АА*
Tshape0
\
Reshape_145/shapeConst*
_output_shapes
:*
valueB:А*
dtype0
i
Reshape_145Reshape
split_3:10Reshape_145/shape*
T0*
Tshape0*
_output_shapes	
:А
b
Reshape_146/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
m
Reshape_146Reshape
split_3:11Reshape_146/shape*
T0*
Tshape0*
_output_shapes
:	А
[
Reshape_147/shapeConst*
valueB:*
_output_shapes
:*
dtype0
h
Reshape_147Reshape
split_3:12Reshape_147/shape*
_output_shapes
:*
T0*
Tshape0
b
Reshape_148/shapeConst*
valueB"<      *
dtype0*
_output_shapes
:
m
Reshape_148Reshape
split_3:13Reshape_148/shape*
T0*
Tshape0*
_output_shapes
:	<А
\
Reshape_149/shapeConst*
_output_shapes
:*
dtype0*
valueB:А
i
Reshape_149Reshape
split_3:14Reshape_149/shape*
T0*
_output_shapes	
:А*
Tshape0
b
Reshape_150/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
n
Reshape_150Reshape
split_3:15Reshape_150/shape*
T0*
Tshape0* 
_output_shapes
:
АА
\
Reshape_151/shapeConst*
valueB:А*
_output_shapes
:*
dtype0
i
Reshape_151Reshape
split_3:16Reshape_151/shape*
Tshape0*
_output_shapes	
:А*
T0
b
Reshape_152/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
m
Reshape_152Reshape
split_3:17Reshape_152/shape*
Tshape0*
T0*
_output_shapes
:	А
[
Reshape_153/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_153Reshape
split_3:18Reshape_153/shape*
T0*
Tshape0*
_output_shapes
:
T
Reshape_154/shapeConst*
_output_shapes
: *
valueB *
dtype0
d
Reshape_154Reshape
split_3:19Reshape_154/shape*
Tshape0*
_output_shapes
: *
T0
T
Reshape_155/shapeConst*
_output_shapes
: *
valueB *
dtype0
d
Reshape_155Reshape
split_3:20Reshape_155/shape*
Tshape0*
_output_shapes
: *
T0
b
Reshape_156/shapeConst*
dtype0*
_output_shapes
:*
valueB"<      
m
Reshape_156Reshape
split_3:21Reshape_156/shape*
Tshape0*
_output_shapes
:	<А*
T0
b
Reshape_157/shapeConst*
dtype0*
valueB"<      *
_output_shapes
:
m
Reshape_157Reshape
split_3:22Reshape_157/shape*
T0*
_output_shapes
:	<А*
Tshape0
\
Reshape_158/shapeConst*
valueB:А*
_output_shapes
:*
dtype0
i
Reshape_158Reshape
split_3:23Reshape_158/shape*
T0*
Tshape0*
_output_shapes	
:А
\
Reshape_159/shapeConst*
_output_shapes
:*
dtype0*
valueB:А
i
Reshape_159Reshape
split_3:24Reshape_159/shape*
Tshape0*
_output_shapes	
:А*
T0
b
Reshape_160/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
n
Reshape_160Reshape
split_3:25Reshape_160/shape* 
_output_shapes
:
АА*
T0*
Tshape0
b
Reshape_161/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
n
Reshape_161Reshape
split_3:26Reshape_161/shape*
Tshape0* 
_output_shapes
:
АА*
T0
\
Reshape_162/shapeConst*
valueB:А*
dtype0*
_output_shapes
:
i
Reshape_162Reshape
split_3:27Reshape_162/shape*
_output_shapes	
:А*
T0*
Tshape0
\
Reshape_163/shapeConst*
valueB:А*
_output_shapes
:*
dtype0
i
Reshape_163Reshape
split_3:28Reshape_163/shape*
_output_shapes	
:А*
T0*
Tshape0
b
Reshape_164/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
m
Reshape_164Reshape
split_3:29Reshape_164/shape*
_output_shapes
:	А*
T0*
Tshape0
b
Reshape_165/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
m
Reshape_165Reshape
split_3:30Reshape_165/shape*
T0*
_output_shapes
:	А*
Tshape0
[
Reshape_166/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_166Reshape
split_3:31Reshape_166/shape*
T0*
_output_shapes
:*
Tshape0
[
Reshape_167/shapeConst*
_output_shapes
:*
dtype0*
valueB:
h
Reshape_167Reshape
split_3:32Reshape_167/shape*
T0*
Tshape0*
_output_shapes
:
b
Reshape_168/shapeConst*
_output_shapes
:*
valueB"<      *
dtype0
m
Reshape_168Reshape
split_3:33Reshape_168/shape*
Tshape0*
T0*
_output_shapes
:	<А
b
Reshape_169/shapeConst*
_output_shapes
:*
valueB"<      *
dtype0
m
Reshape_169Reshape
split_3:34Reshape_169/shape*
Tshape0*
_output_shapes
:	<А*
T0
\
Reshape_170/shapeConst*
_output_shapes
:*
valueB:А*
dtype0
i
Reshape_170Reshape
split_3:35Reshape_170/shape*
Tshape0*
_output_shapes	
:А*
T0
\
Reshape_171/shapeConst*
_output_shapes
:*
dtype0*
valueB:А
i
Reshape_171Reshape
split_3:36Reshape_171/shape*
_output_shapes	
:А*
Tshape0*
T0
b
Reshape_172/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
n
Reshape_172Reshape
split_3:37Reshape_172/shape*
T0* 
_output_shapes
:
АА*
Tshape0
b
Reshape_173/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
n
Reshape_173Reshape
split_3:38Reshape_173/shape*
Tshape0*
T0* 
_output_shapes
:
АА
\
Reshape_174/shapeConst*
dtype0*
valueB:А*
_output_shapes
:
i
Reshape_174Reshape
split_3:39Reshape_174/shape*
T0*
_output_shapes	
:А*
Tshape0
\
Reshape_175/shapeConst*
valueB:А*
_output_shapes
:*
dtype0
i
Reshape_175Reshape
split_3:40Reshape_175/shape*
Tshape0*
_output_shapes	
:А*
T0
b
Reshape_176/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
m
Reshape_176Reshape
split_3:41Reshape_176/shape*
Tshape0*
_output_shapes
:	А*
T0
b
Reshape_177/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
m
Reshape_177Reshape
split_3:42Reshape_177/shape*
Tshape0*
_output_shapes
:	А*
T0
[
Reshape_178/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_178Reshape
split_3:43Reshape_178/shape*
T0*
_output_shapes
:*
Tshape0
[
Reshape_179/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_179Reshape
split_3:44Reshape_179/shape*
T0*
Tshape0*
_output_shapes
:
и
	Assign_19Assignpi/dense/kernelReshape_135*
_output_shapes
:	<А*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
а
	Assign_20Assignpi/dense/biasReshape_136*
validate_shape(*
use_locking(*
_output_shapes	
:А*
T0* 
_class
loc:@pi/dense/bias
н
	Assign_21Assignpi/dense_1/kernelReshape_137*
T0* 
_output_shapes
:
АА*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel
д
	Assign_22Assignpi/dense_1/biasReshape_138*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:А
м
	Assign_23Assignpi/dense_2/kernelReshape_139*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	А*$
_class
loc:@pi/dense_2/kernel
г
	Assign_24Assignpi/dense_2/biasReshape_140*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias
Щ
	Assign_25Assign
pi/log_stdReshape_141*
use_locking(*
_class
loc:@pi/log_std*
_output_shapes
:*
T0*
validate_shape(
и
	Assign_26Assignvf/dense/kernelReshape_142*
validate_shape(*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	<А*
T0
а
	Assign_27Assignvf/dense/biasReshape_143*
use_locking(*
_output_shapes	
:А* 
_class
loc:@vf/dense/bias*
T0*
validate_shape(
н
	Assign_28Assignvf/dense_1/kernelReshape_144*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel
д
	Assign_29Assignvf/dense_1/biasReshape_145*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:А*
T0*
use_locking(
м
	Assign_30Assignvf/dense_2/kernelReshape_146*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	А*$
_class
loc:@vf/dense_2/kernel
г
	Assign_31Assignvf/dense_2/biasReshape_147*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
и
	Assign_32Assignvc/dense/kernelReshape_148*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<А*
validate_shape(*
use_locking(*
T0
а
	Assign_33Assignvc/dense/biasReshape_149*
T0*
_output_shapes	
:А*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias
н
	Assign_34Assignvc/dense_1/kernelReshape_150*
T0* 
_output_shapes
:
АА*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel
д
	Assign_35Assignvc/dense_1/biasReshape_151*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:А*
validate_shape(*
use_locking(
м
	Assign_36Assignvc/dense_2/kernelReshape_152*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	А*
validate_shape(*
T0*
use_locking(
г
	Assign_37Assignvc/dense_2/biasReshape_153*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
Щ
	Assign_38Assignbeta1_powerReshape_154*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
validate_shape(*
T0
Щ
	Assign_39Assignbeta2_powerReshape_155*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes
: 
н
	Assign_40Assignvf/dense/kernel/AdamReshape_156*
T0*
validate_shape(*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	<А
п
	Assign_41Assignvf/dense/kernel/Adam_1Reshape_157*
T0*
_output_shapes
:	<А*
validate_shape(*"
_class
loc:@vf/dense/kernel*
use_locking(
е
	Assign_42Assignvf/dense/bias/AdamReshape_158*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:А
з
	Assign_43Assignvf/dense/bias/Adam_1Reshape_159*
validate_shape(*
_output_shapes	
:А* 
_class
loc:@vf/dense/bias*
T0*
use_locking(
▓
	Assign_44Assignvf/dense_1/kernel/AdamReshape_160* 
_output_shapes
:
АА*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(
┤
	Assign_45Assignvf/dense_1/kernel/Adam_1Reshape_161*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:
АА
й
	Assign_46Assignvf/dense_1/bias/AdamReshape_162*
_output_shapes	
:А*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
use_locking(*
T0
л
	Assign_47Assignvf/dense_1/bias/Adam_1Reshape_163*
validate_shape(*
_output_shapes	
:А*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(
▒
	Assign_48Assignvf/dense_2/kernel/AdamReshape_164*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	А
│
	Assign_49Assignvf/dense_2/kernel/Adam_1Reshape_165*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	А
и
	Assign_50Assignvf/dense_2/bias/AdamReshape_166*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0
к
	Assign_51Assignvf/dense_2/bias/Adam_1Reshape_167*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
н
	Assign_52Assignvc/dense/kernel/AdamReshape_168*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<А*
validate_shape(
п
	Assign_53Assignvc/dense/kernel/Adam_1Reshape_169*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	<А
е
	Assign_54Assignvc/dense/bias/AdamReshape_170* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
_output_shapes	
:А*
validate_shape(
з
	Assign_55Assignvc/dense/bias/Adam_1Reshape_171*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:А* 
_class
loc:@vc/dense/bias
▓
	Assign_56Assignvc/dense_1/kernel/AdamReshape_172*
validate_shape(* 
_output_shapes
:
АА*
T0*
use_locking(*$
_class
loc:@vc/dense_1/kernel
┤
	Assign_57Assignvc/dense_1/kernel/Adam_1Reshape_173* 
_output_shapes
:
АА*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
й
	Assign_58Assignvc/dense_1/bias/AdamReshape_174*
_output_shapes	
:А*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias
л
	Assign_59Assignvc/dense_1/bias/Adam_1Reshape_175*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:А*
use_locking(*
T0*
validate_shape(
▒
	Assign_60Assignvc/dense_2/kernel/AdamReshape_176*
validate_shape(*
T0*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel*
use_locking(
│
	Assign_61Assignvc/dense_2/kernel/Adam_1Reshape_177*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	А
и
	Assign_62Assignvc/dense_2/bias/AdamReshape_178*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
к
	Assign_63Assignvc/dense_2/bias/Adam_1Reshape_179*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias
░
group_deps_3NoOp
^Assign_19
^Assign_20
^Assign_21
^Assign_22
^Assign_23
^Assign_24
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35
^Assign_36
^Assign_37
^Assign_38
^Assign_39
^Assign_40
^Assign_41
^Assign_42
^Assign_43
^Assign_44
^Assign_45
^Assign_46
^Assign_47
^Assign_48
^Assign_49
^Assign_50
^Assign_51
^Assign_52
^Assign_53
^Assign_54
^Assign_55
^Assign_56
^Assign_57
^Assign_58
^Assign_59
^Assign_60
^Assign_61
^Assign_62
^Assign_63
Y
save/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
_output_shapes
: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
Д
save/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_501b93f7d9534e3e9dd6dc0a7988fa36/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Ё
save/SaveV2/tensor_namesConst*
_output_shapes
:-*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
╜
save/SaveV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-*
dtype0
о
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
_output_shapes
: *'
_class
loc:@save/ShardedFilename*
T0
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*

axis *
T0*
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
_output_shapes
: *
T0
є
save/RestoreV2/tensor_namesConst*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
└
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
я
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*;
dtypes1
/2-*╩
_output_shapes╖
┤:::::::::::::::::::::::::::::::::::::::::::::
Ю
save/AssignAssignbeta1_powersave/RestoreV2*
_output_shapes
: *
T0*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias
в
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
_output_shapes
: *
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(
й
save/Assign_2Assignpi/dense/biassave/RestoreV2:2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:А
▒
save/Assign_3Assignpi/dense/kernelsave/RestoreV2:3*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	<А
н
save/Assign_4Assignpi/dense_1/biassave/RestoreV2:4*
use_locking(*
_output_shapes	
:А*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias
╢
save/Assign_5Assignpi/dense_1/kernelsave/RestoreV2:5* 
_output_shapes
:
АА*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel
м
save/Assign_6Assignpi/dense_2/biassave/RestoreV2:6*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(
╡
save/Assign_7Assignpi/dense_2/kernelsave/RestoreV2:7*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	А
в
save/Assign_8Assign
pi/log_stdsave/RestoreV2:8*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
й
save/Assign_9Assignvc/dense/biassave/RestoreV2:9*
_output_shapes	
:А*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(
░
save/Assign_10Assignvc/dense/bias/Adamsave/RestoreV2:10*
use_locking(*
T0*
_output_shapes	
:А*
validate_shape(* 
_class
loc:@vc/dense/bias
▓
save/Assign_11Assignvc/dense/bias/Adam_1save/RestoreV2:11*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:А
│
save/Assign_12Assignvc/dense/kernelsave/RestoreV2:12*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<А*
T0*
use_locking(*
validate_shape(
╕
save/Assign_13Assignvc/dense/kernel/Adamsave/RestoreV2:13*
_output_shapes
:	<А*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense/kernel
║
save/Assign_14Assignvc/dense/kernel/Adam_1save/RestoreV2:14*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<А
п
save/Assign_15Assignvc/dense_1/biassave/RestoreV2:15*
_output_shapes	
:А*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0*
use_locking(
┤
save/Assign_16Assignvc/dense_1/bias/Adamsave/RestoreV2:16*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:А*
use_locking(*
validate_shape(
╢
save/Assign_17Assignvc/dense_1/bias/Adam_1save/RestoreV2:17*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:А
╕
save/Assign_18Assignvc/dense_1/kernelsave/RestoreV2:18* 
_output_shapes
:
АА*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel
╜
save/Assign_19Assignvc/dense_1/kernel/Adamsave/RestoreV2:19*
validate_shape(* 
_output_shapes
:
АА*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
T0
┐
save/Assign_20Assignvc/dense_1/kernel/Adam_1save/RestoreV2:20*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(*
use_locking(* 
_output_shapes
:
АА
о
save/Assign_21Assignvc/dense_2/biassave/RestoreV2:21*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias
│
save/Assign_22Assignvc/dense_2/bias/Adamsave/RestoreV2:22*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:*
T0
╡
save/Assign_23Assignvc/dense_2/bias/Adam_1save/RestoreV2:23*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:*
use_locking(
╖
save/Assign_24Assignvc/dense_2/kernelsave/RestoreV2:24*
validate_shape(*
use_locking(*
_output_shapes
:	А*
T0*$
_class
loc:@vc/dense_2/kernel
╝
save/Assign_25Assignvc/dense_2/kernel/Adamsave/RestoreV2:25*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	А*
use_locking(
╛
save/Assign_26Assignvc/dense_2/kernel/Adam_1save/RestoreV2:26*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0
л
save/Assign_27Assignvf/dense/biassave/RestoreV2:27*
_output_shapes	
:А* 
_class
loc:@vf/dense/bias*
use_locking(*
T0*
validate_shape(
░
save/Assign_28Assignvf/dense/bias/Adamsave/RestoreV2:28*
_output_shapes	
:А*
use_locking(* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0
▓
save/Assign_29Assignvf/dense/bias/Adam_1save/RestoreV2:29*
_output_shapes	
:А*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
use_locking(
│
save/Assign_30Assignvf/dense/kernelsave/RestoreV2:30*
use_locking(*
T0*
_output_shapes
:	<А*
validate_shape(*"
_class
loc:@vf/dense/kernel
╕
save/Assign_31Assignvf/dense/kernel/Adamsave/RestoreV2:31*
_output_shapes
:	<А*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
T0
║
save/Assign_32Assignvf/dense/kernel/Adam_1save/RestoreV2:32*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	<А
п
save/Assign_33Assignvf/dense_1/biassave/RestoreV2:33*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:А*"
_class
loc:@vf/dense_1/bias
┤
save/Assign_34Assignvf/dense_1/bias/Adamsave/RestoreV2:34*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:А*
T0*
validate_shape(*
use_locking(
╢
save/Assign_35Assignvf/dense_1/bias/Adam_1save/RestoreV2:35*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:А*
T0*
use_locking(*
validate_shape(
╕
save/Assign_36Assignvf/dense_1/kernelsave/RestoreV2:36*
use_locking(* 
_output_shapes
:
АА*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(
╜
save/Assign_37Assignvf/dense_1/kernel/Adamsave/RestoreV2:37*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:
АА*
T0
┐
save/Assign_38Assignvf/dense_1/kernel/Adam_1save/RestoreV2:38*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:
АА
о
save/Assign_39Assignvf/dense_2/biassave/RestoreV2:39*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_2/bias
│
save/Assign_40Assignvf/dense_2/bias/Adamsave/RestoreV2:40*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias
╡
save/Assign_41Assignvf/dense_2/bias/Adam_1save/RestoreV2:41*"
_class
loc:@vf/dense_2/bias*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
╖
save/Assign_42Assignvf/dense_2/kernelsave/RestoreV2:42*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	А*
use_locking(*
T0*
validate_shape(
╝
save/Assign_43Assignvf/dense_2/kernel/Adamsave/RestoreV2:43*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	А*
validate_shape(
╛
save/Assign_44Assignvf/dense_2/kernel/Adam_1save/RestoreV2:44*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	А
Л
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
_output_shapes
: *
shape: *
dtype0
Ж
save_1/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b40d8f8ad5e84e00b5b7fdc2c40141bb/part
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
^
save_1/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
Е
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
Є
save_1/SaveV2/tensor_namesConst*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
┐
save_1/SaveV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
╢
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-
Щ
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
г
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
T0*
N*

axis *
_output_shapes
:
Г
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
В
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
ї
save_1/RestoreV2/tensor_namesConst*
dtype0*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-
┬
!save_1/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ў
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*;
dtypes1
/2-*╩
_output_shapes╖
┤:::::::::::::::::::::::::::::::::::::::::::::
в
save_1/AssignAssignbeta1_powersave_1/RestoreV2* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
ж
save_1/Assign_1Assignbeta2_powersave_1/RestoreV2:1*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
validate_shape(
н
save_1/Assign_2Assignpi/dense/biassave_1/RestoreV2:2*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:А*
use_locking(
╡
save_1/Assign_3Assignpi/dense/kernelsave_1/RestoreV2:3*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	<А
▒
save_1/Assign_4Assignpi/dense_1/biassave_1/RestoreV2:4*
_output_shapes	
:А*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0
║
save_1/Assign_5Assignpi/dense_1/kernelsave_1/RestoreV2:5*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
АА*
use_locking(
░
save_1/Assign_6Assignpi/dense_2/biassave_1/RestoreV2:6*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
╣
save_1/Assign_7Assignpi/dense_2/kernelsave_1/RestoreV2:7*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	А*
validate_shape(
ж
save_1/Assign_8Assign
pi/log_stdsave_1/RestoreV2:8*
T0*
_output_shapes
:*
validate_shape(*
_class
loc:@pi/log_std*
use_locking(
н
save_1/Assign_9Assignvc/dense/biassave_1/RestoreV2:9*
use_locking(*
T0*
_output_shapes	
:А*
validate_shape(* 
_class
loc:@vc/dense/bias
┤
save_1/Assign_10Assignvc/dense/bias/Adamsave_1/RestoreV2:10*
validate_shape(*
_output_shapes	
:А* 
_class
loc:@vc/dense/bias*
use_locking(*
T0
╢
save_1/Assign_11Assignvc/dense/bias/Adam_1save_1/RestoreV2:11*
T0*
use_locking(*
_output_shapes	
:А*
validate_shape(* 
_class
loc:@vc/dense/bias
╖
save_1/Assign_12Assignvc/dense/kernelsave_1/RestoreV2:12*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<А*
validate_shape(
╝
save_1/Assign_13Assignvc/dense/kernel/Adamsave_1/RestoreV2:13*"
_class
loc:@vc/dense/kernel*
use_locking(*
_output_shapes
:	<А*
validate_shape(*
T0
╛
save_1/Assign_14Assignvc/dense/kernel/Adam_1save_1/RestoreV2:14*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	<А
│
save_1/Assign_15Assignvc/dense_1/biassave_1/RestoreV2:15*
_output_shapes	
:А*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias
╕
save_1/Assign_16Assignvc/dense_1/bias/Adamsave_1/RestoreV2:16*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:А
║
save_1/Assign_17Assignvc/dense_1/bias/Adam_1save_1/RestoreV2:17*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:А*
use_locking(
╝
save_1/Assign_18Assignvc/dense_1/kernelsave_1/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
АА*
validate_shape(*
use_locking(*
T0
┴
save_1/Assign_19Assignvc/dense_1/kernel/Adamsave_1/RestoreV2:19*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
АА
├
save_1/Assign_20Assignvc/dense_1/kernel/Adam_1save_1/RestoreV2:20* 
_output_shapes
:
АА*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
▓
save_1/Assign_21Assignvc/dense_2/biassave_1/RestoreV2:21*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(
╖
save_1/Assign_22Assignvc/dense_2/bias/Adamsave_1/RestoreV2:22*
_output_shapes
:*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0*
validate_shape(
╣
save_1/Assign_23Assignvc/dense_2/bias/Adam_1save_1/RestoreV2:23*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(
╗
save_1/Assign_24Assignvc/dense_2/kernelsave_1/RestoreV2:24*
use_locking(*
T0*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
└
save_1/Assign_25Assignvc/dense_2/kernel/Adamsave_1/RestoreV2:25*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	А*
T0*
use_locking(
┬
save_1/Assign_26Assignvc/dense_2/kernel/Adam_1save_1/RestoreV2:26*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	А*
use_locking(
п
save_1/Assign_27Assignvf/dense/biassave_1/RestoreV2:27*
_output_shapes	
:А*
validate_shape(*
T0* 
_class
loc:@vf/dense/bias*
use_locking(
┤
save_1/Assign_28Assignvf/dense/bias/Adamsave_1/RestoreV2:28*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:А*
T0
╢
save_1/Assign_29Assignvf/dense/bias/Adam_1save_1/RestoreV2:29* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:А*
T0
╖
save_1/Assign_30Assignvf/dense/kernelsave_1/RestoreV2:30*
validate_shape(*
_output_shapes
:	<А*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel
╝
save_1/Assign_31Assignvf/dense/kernel/Adamsave_1/RestoreV2:31*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	<А
╛
save_1/Assign_32Assignvf/dense/kernel/Adam_1save_1/RestoreV2:32*
_output_shapes
:	<А*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(
│
save_1/Assign_33Assignvf/dense_1/biassave_1/RestoreV2:33*
_output_shapes	
:А*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(
╕
save_1/Assign_34Assignvf/dense_1/bias/Adamsave_1/RestoreV2:34*
T0*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:А
║
save_1/Assign_35Assignvf/dense_1/bias/Adam_1save_1/RestoreV2:35*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:А*
use_locking(*
T0*
validate_shape(
╝
save_1/Assign_36Assignvf/dense_1/kernelsave_1/RestoreV2:36*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
АА*
use_locking(*
validate_shape(
┴
save_1/Assign_37Assignvf/dense_1/kernel/Adamsave_1/RestoreV2:37*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:
АА*
T0
├
save_1/Assign_38Assignvf/dense_1/kernel/Adam_1save_1/RestoreV2:38*
use_locking(* 
_output_shapes
:
АА*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel
▓
save_1/Assign_39Assignvf/dense_2/biassave_1/RestoreV2:39*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
use_locking(
╖
save_1/Assign_40Assignvf/dense_2/bias/Adamsave_1/RestoreV2:40*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias
╣
save_1/Assign_41Assignvf/dense_2/bias/Adam_1save_1/RestoreV2:41*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
╗
save_1/Assign_42Assignvf/dense_2/kernelsave_1/RestoreV2:42*
_output_shapes
:	А*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(*
T0
└
save_1/Assign_43Assignvf/dense_2/kernel/Adamsave_1/RestoreV2:43*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	А*$
_class
loc:@vf/dense_2/kernel
┬
save_1/Assign_44Assignvf/dense_2/kernel/Adam_1save_1/RestoreV2:44*
_output_shapes
:	А*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
ч
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
_output_shapes
: *
dtype0*
shape: 
Ж
save_2/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2363b6f3535e41529e8881e7102a3b6c/part
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
Е
save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
Є
save_2/SaveV2/tensor_namesConst*
_output_shapes
:-*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
┐
save_2/SaveV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
╢
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-
Щ
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*)
_class
loc:@save_2/ShardedFilename*
T0*
_output_shapes
: 
г
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
N*
T0*
_output_shapes
:*

axis 
Г
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
В
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
T0*
_output_shapes
: 
ї
save_2/RestoreV2/tensor_namesConst*
dtype0*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-
┬
!save_2/RestoreV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
ў
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*;
dtypes1
/2-*╩
_output_shapes╖
┤:::::::::::::::::::::::::::::::::::::::::::::
в
save_2/AssignAssignbeta1_powersave_2/RestoreV2*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(
ж
save_2/Assign_1Assignbeta2_powersave_2/RestoreV2:1*
T0*
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
use_locking(
н
save_2/Assign_2Assignpi/dense/biassave_2/RestoreV2:2*
use_locking(*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:А
╡
save_2/Assign_3Assignpi/dense/kernelsave_2/RestoreV2:3*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	<А*"
_class
loc:@pi/dense/kernel
▒
save_2/Assign_4Assignpi/dense_1/biassave_2/RestoreV2:4*
validate_shape(*
_output_shapes	
:А*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias
║
save_2/Assign_5Assignpi/dense_1/kernelsave_2/RestoreV2:5*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
АА*
use_locking(
░
save_2/Assign_6Assignpi/dense_2/biassave_2/RestoreV2:6*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias
╣
save_2/Assign_7Assignpi/dense_2/kernelsave_2/RestoreV2:7*
use_locking(*
_output_shapes
:	А*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
ж
save_2/Assign_8Assign
pi/log_stdsave_2/RestoreV2:8*
_output_shapes
:*
T0*
validate_shape(*
_class
loc:@pi/log_std*
use_locking(
н
save_2/Assign_9Assignvc/dense/biassave_2/RestoreV2:9*
_output_shapes	
:А*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0
┤
save_2/Assign_10Assignvc/dense/bias/Adamsave_2/RestoreV2:10*
_output_shapes	
:А*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(
╢
save_2/Assign_11Assignvc/dense/bias/Adam_1save_2/RestoreV2:11*
use_locking(*
_output_shapes	
:А* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(
╖
save_2/Assign_12Assignvc/dense/kernelsave_2/RestoreV2:12*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<А
╝
save_2/Assign_13Assignvc/dense/kernel/Adamsave_2/RestoreV2:13*
use_locking(*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	<А
╛
save_2/Assign_14Assignvc/dense/kernel/Adam_1save_2/RestoreV2:14*
T0*
_output_shapes
:	<А*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(
│
save_2/Assign_15Assignvc/dense_1/biassave_2/RestoreV2:15*
_output_shapes	
:А*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias
╕
save_2/Assign_16Assignvc/dense_1/bias/Adamsave_2/RestoreV2:16*
validate_shape(*
_output_shapes	
:А*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0
║
save_2/Assign_17Assignvc/dense_1/bias/Adam_1save_2/RestoreV2:17*"
_class
loc:@vc/dense_1/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:А
╝
save_2/Assign_18Assignvc/dense_1/kernelsave_2/RestoreV2:18*
T0* 
_output_shapes
:
АА*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
┴
save_2/Assign_19Assignvc/dense_1/kernel/Adamsave_2/RestoreV2:19*
validate_shape(*
T0* 
_output_shapes
:
АА*
use_locking(*$
_class
loc:@vc/dense_1/kernel
├
save_2/Assign_20Assignvc/dense_1/kernel/Adam_1save_2/RestoreV2:20* 
_output_shapes
:
АА*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
▓
save_2/Assign_21Assignvc/dense_2/biassave_2/RestoreV2:21*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
╖
save_2/Assign_22Assignvc/dense_2/bias/Adamsave_2/RestoreV2:22*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:
╣
save_2/Assign_23Assignvc/dense_2/bias/Adam_1save_2/RestoreV2:23*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
╗
save_2/Assign_24Assignvc/dense_2/kernelsave_2/RestoreV2:24*
use_locking(*
_output_shapes
:	А*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
└
save_2/Assign_25Assignvc/dense_2/kernel/Adamsave_2/RestoreV2:25*
validate_shape(*
use_locking(*
_output_shapes
:	А*
T0*$
_class
loc:@vc/dense_2/kernel
┬
save_2/Assign_26Assignvc/dense_2/kernel/Adam_1save_2/RestoreV2:26*
validate_shape(*
_output_shapes
:	А*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel
п
save_2/Assign_27Assignvf/dense/biassave_2/RestoreV2:27*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:А
┤
save_2/Assign_28Assignvf/dense/bias/Adamsave_2/RestoreV2:28*
use_locking(*
_output_shapes	
:А*
validate_shape(* 
_class
loc:@vf/dense/bias*
T0
╢
save_2/Assign_29Assignvf/dense/bias/Adam_1save_2/RestoreV2:29*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:А
╖
save_2/Assign_30Assignvf/dense/kernelsave_2/RestoreV2:30*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	<А*"
_class
loc:@vf/dense/kernel
╝
save_2/Assign_31Assignvf/dense/kernel/Adamsave_2/RestoreV2:31*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	<А*
T0
╛
save_2/Assign_32Assignvf/dense/kernel/Adam_1save_2/RestoreV2:32*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	<А*
T0*
validate_shape(
│
save_2/Assign_33Assignvf/dense_1/biassave_2/RestoreV2:33*
_output_shapes	
:А*"
_class
loc:@vf/dense_1/bias*
T0*
validate_shape(*
use_locking(
╕
save_2/Assign_34Assignvf/dense_1/bias/Adamsave_2/RestoreV2:34*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:А
║
save_2/Assign_35Assignvf/dense_1/bias/Adam_1save_2/RestoreV2:35*
_output_shapes	
:А*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(*
validate_shape(
╝
save_2/Assign_36Assignvf/dense_1/kernelsave_2/RestoreV2:36*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:
АА
┴
save_2/Assign_37Assignvf/dense_1/kernel/Adamsave_2/RestoreV2:37*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
АА*
T0
├
save_2/Assign_38Assignvf/dense_1/kernel/Adam_1save_2/RestoreV2:38* 
_output_shapes
:
АА*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(
▓
save_2/Assign_39Assignvf/dense_2/biassave_2/RestoreV2:39*"
_class
loc:@vf/dense_2/bias*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
╖
save_2/Assign_40Assignvf/dense_2/bias/Adamsave_2/RestoreV2:40*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0
╣
save_2/Assign_41Assignvf/dense_2/bias/Adam_1save_2/RestoreV2:41*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
╗
save_2/Assign_42Assignvf/dense_2/kernelsave_2/RestoreV2:42*
_output_shapes
:	А*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
T0
└
save_2/Assign_43Assignvf/dense_2/kernel/Adamsave_2/RestoreV2:43*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	А*
validate_shape(*
T0
┬
save_2/Assign_44Assignvf/dense_2/kernel/Adam_1save_2/RestoreV2:44*
T0*
_output_shapes
:	А*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
ч
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_40^save_2/Assign_41^save_2/Assign_42^save_2/Assign_43^save_2/Assign_44^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
_output_shapes
: *
shape: *
dtype0
Ж
save_3/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2373f9a7e7a64c1cad5f19e671b18294/part
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
S
save_3/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
^
save_3/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
Е
save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
Є
save_3/SaveV2/tensor_namesConst*
_output_shapes
:-*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
┐
save_3/SaveV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
╢
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-
Щ
save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: *
T0
г
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*

axis *
_output_shapes
:*
T0*
N
Г
save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(
В
save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
_output_shapes
: *
T0
ї
save_3/RestoreV2/tensor_namesConst*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
┬
!save_3/RestoreV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-*
dtype0
ў
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*╩
_output_shapes╖
┤:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
в
save_3/AssignAssignbeta1_powersave_3/RestoreV2*
use_locking(*
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0
ж
save_3/Assign_1Assignbeta2_powersave_3/RestoreV2:1*
_output_shapes
: *
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
use_locking(
н
save_3/Assign_2Assignpi/dense/biassave_3/RestoreV2:2*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
╡
save_3/Assign_3Assignpi/dense/kernelsave_3/RestoreV2:3*
T0*
validate_shape(*
_output_shapes
:	<А*
use_locking(*"
_class
loc:@pi/dense/kernel
▒
save_3/Assign_4Assignpi/dense_1/biassave_3/RestoreV2:4*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes	
:А
║
save_3/Assign_5Assignpi/dense_1/kernelsave_3/RestoreV2:5*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
АА*
T0*
validate_shape(
░
save_3/Assign_6Assignpi/dense_2/biassave_3/RestoreV2:6*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(
╣
save_3/Assign_7Assignpi/dense_2/kernelsave_3/RestoreV2:7*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	А
ж
save_3/Assign_8Assign
pi/log_stdsave_3/RestoreV2:8*
_class
loc:@pi/log_std*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
н
save_3/Assign_9Assignvc/dense/biassave_3/RestoreV2:9*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:А* 
_class
loc:@vc/dense/bias
┤
save_3/Assign_10Assignvc/dense/bias/Adamsave_3/RestoreV2:10*
_output_shapes	
:А*
T0* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(
╢
save_3/Assign_11Assignvc/dense/bias/Adam_1save_3/RestoreV2:11* 
_class
loc:@vc/dense/bias*
use_locking(*
T0*
_output_shapes	
:А*
validate_shape(
╖
save_3/Assign_12Assignvc/dense/kernelsave_3/RestoreV2:12*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	<А*
T0
╝
save_3/Assign_13Assignvc/dense/kernel/Adamsave_3/RestoreV2:13*
T0*
validate_shape(*
_output_shapes
:	<А*"
_class
loc:@vc/dense/kernel*
use_locking(
╛
save_3/Assign_14Assignvc/dense/kernel/Adam_1save_3/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
use_locking(*
_output_shapes
:	<А*
T0*
validate_shape(
│
save_3/Assign_15Assignvc/dense_1/biassave_3/RestoreV2:15*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias
╕
save_3/Assign_16Assignvc/dense_1/bias/Adamsave_3/RestoreV2:16*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:А*
validate_shape(
║
save_3/Assign_17Assignvc/dense_1/bias/Adam_1save_3/RestoreV2:17*
use_locking(*
_output_shapes	
:А*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0
╝
save_3/Assign_18Assignvc/dense_1/kernelsave_3/RestoreV2:18*
validate_shape(*
use_locking(* 
_output_shapes
:
АА*$
_class
loc:@vc/dense_1/kernel*
T0
┴
save_3/Assign_19Assignvc/dense_1/kernel/Adamsave_3/RestoreV2:19*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
АА*
validate_shape(
├
save_3/Assign_20Assignvc/dense_1/kernel/Adam_1save_3/RestoreV2:20*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
АА
▓
save_3/Assign_21Assignvc/dense_2/biassave_3/RestoreV2:21*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias
╖
save_3/Assign_22Assignvc/dense_2/bias/Adamsave_3/RestoreV2:22*
T0*
use_locking(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(
╣
save_3/Assign_23Assignvc/dense_2/bias/Adam_1save_3/RestoreV2:23*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias
╗
save_3/Assign_24Assignvc/dense_2/kernelsave_3/RestoreV2:24*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	А
└
save_3/Assign_25Assignvc/dense_2/kernel/Adamsave_3/RestoreV2:25*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	А*
validate_shape(*
T0*
use_locking(
┬
save_3/Assign_26Assignvc/dense_2/kernel/Adam_1save_3/RestoreV2:26*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	А*
T0
п
save_3/Assign_27Assignvf/dense/biassave_3/RestoreV2:27* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:А*
T0*
use_locking(
┤
save_3/Assign_28Assignvf/dense/bias/Adamsave_3/RestoreV2:28*
validate_shape(*
T0*
_output_shapes	
:А* 
_class
loc:@vf/dense/bias*
use_locking(
╢
save_3/Assign_29Assignvf/dense/bias/Adam_1save_3/RestoreV2:29*
_output_shapes	
:А*
use_locking(*
validate_shape(*
T0* 
_class
loc:@vf/dense/bias
╖
save_3/Assign_30Assignvf/dense/kernelsave_3/RestoreV2:30*
T0*
_output_shapes
:	<А*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(
╝
save_3/Assign_31Assignvf/dense/kernel/Adamsave_3/RestoreV2:31*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<А
╛
save_3/Assign_32Assignvf/dense/kernel/Adam_1save_3/RestoreV2:32*
validate_shape(*
T0*
_output_shapes
:	<А*"
_class
loc:@vf/dense/kernel*
use_locking(
│
save_3/Assign_33Assignvf/dense_1/biassave_3/RestoreV2:33*
validate_shape(*
_output_shapes	
:А*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(
╕
save_3/Assign_34Assignvf/dense_1/bias/Adamsave_3/RestoreV2:34*
T0*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:А
║
save_3/Assign_35Assignvf/dense_1/bias/Adam_1save_3/RestoreV2:35*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:А
╝
save_3/Assign_36Assignvf/dense_1/kernelsave_3/RestoreV2:36* 
_output_shapes
:
АА*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(*
T0
┴
save_3/Assign_37Assignvf/dense_1/kernel/Adamsave_3/RestoreV2:37*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:
АА
├
save_3/Assign_38Assignvf/dense_1/kernel/Adam_1save_3/RestoreV2:38* 
_output_shapes
:
АА*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(
▓
save_3/Assign_39Assignvf/dense_2/biassave_3/RestoreV2:39*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
╖
save_3/Assign_40Assignvf/dense_2/bias/Adamsave_3/RestoreV2:40*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_2/bias
╣
save_3/Assign_41Assignvf/dense_2/bias/Adam_1save_3/RestoreV2:41*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias
╗
save_3/Assign_42Assignvf/dense_2/kernelsave_3/RestoreV2:42*
_output_shapes
:	А*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
└
save_3/Assign_43Assignvf/dense_2/kernel/Adamsave_3/RestoreV2:43*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	А
┬
save_3/Assign_44Assignvf/dense_2/kernel/Adam_1save_3/RestoreV2:44*
use_locking(*
_output_shapes
:	А*$
_class
loc:@vf/dense_2/kernel*
T0*
validate_shape(
ч
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_5^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard
[
save_4/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
shape: *
dtype0*
_output_shapes
: 
Ж
save_4/StringJoin/inputs_1Const*<
value3B1 B+_temp_480a9ff719f1460e9b84f837d082b924/part*
_output_shapes
: *
dtype0
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_4/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_4/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
Є
save_4/SaveV2/tensor_namesConst*
dtype0*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-
┐
save_4/SaveV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
╢
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-
Щ
save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_4/ShardedFilename
г
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*
N*
T0*
_output_shapes
:*

axis 
Г
save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(
В
save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
_output_shapes
: *
T0
ї
save_4/RestoreV2/tensor_namesConst*
dtype0*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-
┬
!save_4/RestoreV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-*
dtype0
ў
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*╩
_output_shapes╖
┤:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
в
save_4/AssignAssignbeta1_powersave_4/RestoreV2*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: *
use_locking(
ж
save_4/Assign_1Assignbeta2_powersave_4/RestoreV2:1* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0*
_output_shapes
: *
use_locking(
н
save_4/Assign_2Assignpi/dense/biassave_4/RestoreV2:2* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
╡
save_4/Assign_3Assignpi/dense/kernelsave_4/RestoreV2:3*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	<А*
T0
▒
save_4/Assign_4Assignpi/dense_1/biassave_4/RestoreV2:4*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:А*
use_locking(*
T0*
validate_shape(
║
save_4/Assign_5Assignpi/dense_1/kernelsave_4/RestoreV2:5*
validate_shape(* 
_output_shapes
:
АА*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0
░
save_4/Assign_6Assignpi/dense_2/biassave_4/RestoreV2:6*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
╣
save_4/Assign_7Assignpi/dense_2/kernelsave_4/RestoreV2:7*
_output_shapes
:	А*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(*
T0
ж
save_4/Assign_8Assign
pi/log_stdsave_4/RestoreV2:8*
use_locking(*
_class
loc:@pi/log_std*
T0*
_output_shapes
:*
validate_shape(
н
save_4/Assign_9Assignvc/dense/biassave_4/RestoreV2:9* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:А
┤
save_4/Assign_10Assignvc/dense/bias/Adamsave_4/RestoreV2:10*
use_locking(*
validate_shape(*
_output_shapes	
:А*
T0* 
_class
loc:@vc/dense/bias
╢
save_4/Assign_11Assignvc/dense/bias/Adam_1save_4/RestoreV2:11*
T0*
validate_shape(*
_output_shapes	
:А* 
_class
loc:@vc/dense/bias*
use_locking(
╖
save_4/Assign_12Assignvc/dense/kernelsave_4/RestoreV2:12*
_output_shapes
:	<А*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0
╝
save_4/Assign_13Assignvc/dense/kernel/Adamsave_4/RestoreV2:13*
_output_shapes
:	<А*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel
╛
save_4/Assign_14Assignvc/dense/kernel/Adam_1save_4/RestoreV2:14*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	<А*"
_class
loc:@vc/dense/kernel
│
save_4/Assign_15Assignvc/dense_1/biassave_4/RestoreV2:15*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:А*
use_locking(*
validate_shape(
╕
save_4/Assign_16Assignvc/dense_1/bias/Adamsave_4/RestoreV2:16*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:А*
use_locking(*
T0*
validate_shape(
║
save_4/Assign_17Assignvc/dense_1/bias/Adam_1save_4/RestoreV2:17*
use_locking(*
_output_shapes	
:А*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0
╝
save_4/Assign_18Assignvc/dense_1/kernelsave_4/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
АА
┴
save_4/Assign_19Assignvc/dense_1/kernel/Adamsave_4/RestoreV2:19*
use_locking(* 
_output_shapes
:
АА*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
├
save_4/Assign_20Assignvc/dense_1/kernel/Adam_1save_4/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:
АА
▓
save_4/Assign_21Assignvc/dense_2/biassave_4/RestoreV2:21*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@vc/dense_2/bias
╖
save_4/Assign_22Assignvc/dense_2/bias/Adamsave_4/RestoreV2:22*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
use_locking(
╣
save_4/Assign_23Assignvc/dense_2/bias/Adam_1save_4/RestoreV2:23*
T0*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:
╗
save_4/Assign_24Assignvc/dense_2/kernelsave_4/RestoreV2:24*
T0*
_output_shapes
:	А*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
└
save_4/Assign_25Assignvc/dense_2/kernel/Adamsave_4/RestoreV2:25*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	А*
T0
┬
save_4/Assign_26Assignvc/dense_2/kernel/Adam_1save_4/RestoreV2:26*
use_locking(*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0
п
save_4/Assign_27Assignvf/dense/biassave_4/RestoreV2:27*
validate_shape(*
use_locking(*
_output_shapes	
:А* 
_class
loc:@vf/dense/bias*
T0
┤
save_4/Assign_28Assignvf/dense/bias/Adamsave_4/RestoreV2:28*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:А* 
_class
loc:@vf/dense/bias
╢
save_4/Assign_29Assignvf/dense/bias/Adam_1save_4/RestoreV2:29* 
_class
loc:@vf/dense/bias*
_output_shapes	
:А*
validate_shape(*
use_locking(*
T0
╖
save_4/Assign_30Assignvf/dense/kernelsave_4/RestoreV2:30*
use_locking(*
T0*
_output_shapes
:	<А*"
_class
loc:@vf/dense/kernel*
validate_shape(
╝
save_4/Assign_31Assignvf/dense/kernel/Adamsave_4/RestoreV2:31*
T0*
_output_shapes
:	<А*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(
╛
save_4/Assign_32Assignvf/dense/kernel/Adam_1save_4/RestoreV2:32*
T0*
validate_shape(*
_output_shapes
:	<А*
use_locking(*"
_class
loc:@vf/dense/kernel
│
save_4/Assign_33Assignvf/dense_1/biassave_4/RestoreV2:33*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:А*
use_locking(*
T0*
validate_shape(
╕
save_4/Assign_34Assignvf/dense_1/bias/Adamsave_4/RestoreV2:34*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(*
_output_shapes	
:А
║
save_4/Assign_35Assignvf/dense_1/bias/Adam_1save_4/RestoreV2:35*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:А
╝
save_4/Assign_36Assignvf/dense_1/kernelsave_4/RestoreV2:36*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:
АА
┴
save_4/Assign_37Assignvf/dense_1/kernel/Adamsave_4/RestoreV2:37* 
_output_shapes
:
АА*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(
├
save_4/Assign_38Assignvf/dense_1/kernel/Adam_1save_4/RestoreV2:38* 
_output_shapes
:
АА*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel
▓
save_4/Assign_39Assignvf/dense_2/biassave_4/RestoreV2:39*
use_locking(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0
╖
save_4/Assign_40Assignvf/dense_2/bias/Adamsave_4/RestoreV2:40*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
╣
save_4/Assign_41Assignvf/dense_2/bias/Adam_1save_4/RestoreV2:41*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
╗
save_4/Assign_42Assignvf/dense_2/kernelsave_4/RestoreV2:42*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel
└
save_4/Assign_43Assignvf/dense_2/kernel/Adamsave_4/RestoreV2:43*
_output_shapes
:	А*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(*
T0
┬
save_4/Assign_44Assignvf/dense_2/kernel/Adam_1save_4/RestoreV2:44*
use_locking(*
validate_shape(*
_output_shapes
:	А*$
_class
loc:@vf/dense_2/kernel*
T0
ч
save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_2^save_4/Assign_20^save_4/Assign_21^save_4/Assign_22^save_4/Assign_23^save_4/Assign_24^save_4/Assign_25^save_4/Assign_26^save_4/Assign_27^save_4/Assign_28^save_4/Assign_29^save_4/Assign_3^save_4/Assign_30^save_4/Assign_31^save_4/Assign_32^save_4/Assign_33^save_4/Assign_34^save_4/Assign_35^save_4/Assign_36^save_4/Assign_37^save_4/Assign_38^save_4/Assign_39^save_4/Assign_4^save_4/Assign_40^save_4/Assign_41^save_4/Assign_42^save_4/Assign_43^save_4/Assign_44^save_4/Assign_5^save_4/Assign_6^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard
[
save_5/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
shape: *
_output_shapes
: *
dtype0
Ж
save_5/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_fb594b186364421bafb9845b2390fd0c/part
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
S
save_5/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_5/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
Е
save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
Є
save_5/SaveV2/tensor_namesConst*
_output_shapes
:-*
dtype0*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
┐
save_5/SaveV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
╢
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-
Щ
save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_5/ShardedFilename
г
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*
N*

axis *
_output_shapes
:*
T0
Г
save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(
В
save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
T0*
_output_shapes
: 
ї
save_5/RestoreV2/tensor_namesConst*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
┬
!save_5/RestoreV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
ў
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*╩
_output_shapes╖
┤:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
в
save_5/AssignAssignbeta1_powersave_5/RestoreV2*
validate_shape(*
use_locking(*
T0*
_output_shapes
: * 
_class
loc:@vc/dense/bias
ж
save_5/Assign_1Assignbeta2_powersave_5/RestoreV2:1*
T0*
_output_shapes
: *
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias
н
save_5/Assign_2Assignpi/dense/biassave_5/RestoreV2:2*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes	
:А
╡
save_5/Assign_3Assignpi/dense/kernelsave_5/RestoreV2:3*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	<А*"
_class
loc:@pi/dense/kernel
▒
save_5/Assign_4Assignpi/dense_1/biassave_5/RestoreV2:4*
use_locking(*
validate_shape(*
_output_shapes	
:А*
T0*"
_class
loc:@pi/dense_1/bias
║
save_5/Assign_5Assignpi/dense_1/kernelsave_5/RestoreV2:5*
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
АА
░
save_5/Assign_6Assignpi/dense_2/biassave_5/RestoreV2:6*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
╣
save_5/Assign_7Assignpi/dense_2/kernelsave_5/RestoreV2:7*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes
:	А*
validate_shape(*
T0
ж
save_5/Assign_8Assign
pi/log_stdsave_5/RestoreV2:8*
validate_shape(*
_output_shapes
:*
use_locking(*
_class
loc:@pi/log_std*
T0
н
save_5/Assign_9Assignvc/dense/biassave_5/RestoreV2:9*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:А* 
_class
loc:@vc/dense/bias
┤
save_5/Assign_10Assignvc/dense/bias/Adamsave_5/RestoreV2:10*
_output_shapes	
:А*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(
╢
save_5/Assign_11Assignvc/dense/bias/Adam_1save_5/RestoreV2:11*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:А
╖
save_5/Assign_12Assignvc/dense/kernelsave_5/RestoreV2:12*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(*
_output_shapes
:	<А*
validate_shape(
╝
save_5/Assign_13Assignvc/dense/kernel/Adamsave_5/RestoreV2:13*
T0*
_output_shapes
:	<А*
validate_shape(*"
_class
loc:@vc/dense/kernel*
use_locking(
╛
save_5/Assign_14Assignvc/dense/kernel/Adam_1save_5/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<А*
T0*
validate_shape(*
use_locking(
│
save_5/Assign_15Assignvc/dense_1/biassave_5/RestoreV2:15*
T0*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:А
╕
save_5/Assign_16Assignvc/dense_1/bias/Adamsave_5/RestoreV2:16*
_output_shapes	
:А*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0
║
save_5/Assign_17Assignvc/dense_1/bias/Adam_1save_5/RestoreV2:17*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:А*
validate_shape(
╝
save_5/Assign_18Assignvc/dense_1/kernelsave_5/RestoreV2:18* 
_output_shapes
:
АА*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(
┴
save_5/Assign_19Assignvc/dense_1/kernel/Adamsave_5/RestoreV2:19*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
АА*$
_class
loc:@vc/dense_1/kernel
├
save_5/Assign_20Assignvc/dense_1/kernel/Adam_1save_5/RestoreV2:20*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:
АА*
use_locking(
▓
save_5/Assign_21Assignvc/dense_2/biassave_5/RestoreV2:21*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
╖
save_5/Assign_22Assignvc/dense_2/bias/Adamsave_5/RestoreV2:22*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(
╣
save_5/Assign_23Assignvc/dense_2/bias/Adam_1save_5/RestoreV2:23*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(
╗
save_5/Assign_24Assignvc/dense_2/kernelsave_5/RestoreV2:24*
use_locking(*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel*
T0*
validate_shape(
└
save_5/Assign_25Assignvc/dense_2/kernel/Adamsave_5/RestoreV2:25*
use_locking(*
validate_shape(*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel*
T0
┬
save_5/Assign_26Assignvc/dense_2/kernel/Adam_1save_5/RestoreV2:26*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	А*
use_locking(*
validate_shape(
п
save_5/Assign_27Assignvf/dense/biassave_5/RestoreV2:27*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:А*
T0
┤
save_5/Assign_28Assignvf/dense/bias/Adamsave_5/RestoreV2:28*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:А*
T0
╢
save_5/Assign_29Assignvf/dense/bias/Adam_1save_5/RestoreV2:29*
_output_shapes	
:А* 
_class
loc:@vf/dense/bias*
validate_shape(*
use_locking(*
T0
╖
save_5/Assign_30Assignvf/dense/kernelsave_5/RestoreV2:30*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	<А
╝
save_5/Assign_31Assignvf/dense/kernel/Adamsave_5/RestoreV2:31*
validate_shape(*
_output_shapes
:	<А*"
_class
loc:@vf/dense/kernel*
use_locking(*
T0
╛
save_5/Assign_32Assignvf/dense/kernel/Adam_1save_5/RestoreV2:32*
_output_shapes
:	<А*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(*
use_locking(
│
save_5/Assign_33Assignvf/dense_1/biassave_5/RestoreV2:33*
T0*
use_locking(*
_output_shapes	
:А*
validate_shape(*"
_class
loc:@vf/dense_1/bias
╕
save_5/Assign_34Assignvf/dense_1/bias/Adamsave_5/RestoreV2:34*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:А*
T0*
use_locking(
║
save_5/Assign_35Assignvf/dense_1/bias/Adam_1save_5/RestoreV2:35*
validate_shape(*
_output_shapes	
:А*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(
╝
save_5/Assign_36Assignvf/dense_1/kernelsave_5/RestoreV2:36* 
_output_shapes
:
АА*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel*
use_locking(
┴
save_5/Assign_37Assignvf/dense_1/kernel/Adamsave_5/RestoreV2:37*
validate_shape(*
T0* 
_output_shapes
:
АА*
use_locking(*$
_class
loc:@vf/dense_1/kernel
├
save_5/Assign_38Assignvf/dense_1/kernel/Adam_1save_5/RestoreV2:38*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
АА*$
_class
loc:@vf/dense_1/kernel
▓
save_5/Assign_39Assignvf/dense_2/biassave_5/RestoreV2:39*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias
╖
save_5/Assign_40Assignvf/dense_2/bias/Adamsave_5/RestoreV2:40*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
╣
save_5/Assign_41Assignvf/dense_2/bias/Adam_1save_5/RestoreV2:41*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
╗
save_5/Assign_42Assignvf/dense_2/kernelsave_5/RestoreV2:42*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	А*
use_locking(
└
save_5/Assign_43Assignvf/dense_2/kernel/Adamsave_5/RestoreV2:43*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel
┬
save_5/Assign_44Assignvf/dense_2/kernel/Adam_1save_5/RestoreV2:44*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	А*
validate_shape(*
use_locking(
ч
save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_2^save_5/Assign_20^save_5/Assign_21^save_5/Assign_22^save_5/Assign_23^save_5/Assign_24^save_5/Assign_25^save_5/Assign_26^save_5/Assign_27^save_5/Assign_28^save_5/Assign_29^save_5/Assign_3^save_5/Assign_30^save_5/Assign_31^save_5/Assign_32^save_5/Assign_33^save_5/Assign_34^save_5/Assign_35^save_5/Assign_36^save_5/Assign_37^save_5/Assign_38^save_5/Assign_39^save_5/Assign_4^save_5/Assign_40^save_5/Assign_41^save_5/Assign_42^save_5/Assign_43^save_5/Assign_44^save_5/Assign_5^save_5/Assign_6^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard
[
save_6/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
shape: *
dtype0*
_output_shapes
: 
Ж
save_6/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_da3d24c3d13646ee9174de9ad78b9639/part*
dtype0
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_6/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_6/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
Є
save_6/SaveV2/tensor_namesConst*
dtype0*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-
┐
save_6/SaveV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-*
dtype0
╢
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-
Щ
save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*
T0*)
_class
loc:@save_6/ShardedFilename*
_output_shapes
: 
г
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
N*
_output_shapes
:*
T0*

axis 
Г
save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(
В
save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
_output_shapes
: *
T0
ї
save_6/RestoreV2/tensor_namesConst*
_output_shapes
:-*
dtype0*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
┬
!save_6/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ў
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*╩
_output_shapes╖
┤:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
в
save_6/AssignAssignbeta1_powersave_6/RestoreV2*
T0*
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
use_locking(
ж
save_6/Assign_1Assignbeta2_powersave_6/RestoreV2:1* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: *
validate_shape(*
use_locking(
н
save_6/Assign_2Assignpi/dense/biassave_6/RestoreV2:2*
validate_shape(*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:А
╡
save_6/Assign_3Assignpi/dense/kernelsave_6/RestoreV2:3*
_output_shapes
:	<А*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
use_locking(
▒
save_6/Assign_4Assignpi/dense_1/biassave_6/RestoreV2:4*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes	
:А
║
save_6/Assign_5Assignpi/dense_1/kernelsave_6/RestoreV2:5*
validate_shape(*
use_locking(* 
_output_shapes
:
АА*
T0*$
_class
loc:@pi/dense_1/kernel
░
save_6/Assign_6Assignpi/dense_2/biassave_6/RestoreV2:6*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:
╣
save_6/Assign_7Assignpi/dense_2/kernelsave_6/RestoreV2:7*
_output_shapes
:	А*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
validate_shape(
ж
save_6/Assign_8Assign
pi/log_stdsave_6/RestoreV2:8*
validate_shape(*
_output_shapes
:*
use_locking(*
_class
loc:@pi/log_std*
T0
н
save_6/Assign_9Assignvc/dense/biassave_6/RestoreV2:9*
validate_shape(*
T0*
_output_shapes	
:А*
use_locking(* 
_class
loc:@vc/dense/bias
┤
save_6/Assign_10Assignvc/dense/bias/Adamsave_6/RestoreV2:10*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:А*
validate_shape(
╢
save_6/Assign_11Assignvc/dense/bias/Adam_1save_6/RestoreV2:11*
validate_shape(*
_output_shapes	
:А* 
_class
loc:@vc/dense/bias*
T0*
use_locking(
╖
save_6/Assign_12Assignvc/dense/kernelsave_6/RestoreV2:12*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<А*
validate_shape(
╝
save_6/Assign_13Assignvc/dense/kernel/Adamsave_6/RestoreV2:13*
_output_shapes
:	<А*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(
╛
save_6/Assign_14Assignvc/dense/kernel/Adam_1save_6/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	<А*
T0
│
save_6/Assign_15Assignvc/dense_1/biassave_6/RestoreV2:15*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:А
╕
save_6/Assign_16Assignvc/dense_1/bias/Adamsave_6/RestoreV2:16*"
_class
loc:@vc/dense_1/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:А
║
save_6/Assign_17Assignvc/dense_1/bias/Adam_1save_6/RestoreV2:17*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:А
╝
save_6/Assign_18Assignvc/dense_1/kernelsave_6/RestoreV2:18*
validate_shape(*
use_locking(* 
_output_shapes
:
АА*
T0*$
_class
loc:@vc/dense_1/kernel
┴
save_6/Assign_19Assignvc/dense_1/kernel/Adamsave_6/RestoreV2:19*
validate_shape(* 
_output_shapes
:
АА*
T0*
use_locking(*$
_class
loc:@vc/dense_1/kernel
├
save_6/Assign_20Assignvc/dense_1/kernel/Adam_1save_6/RestoreV2:20* 
_output_shapes
:
АА*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
▓
save_6/Assign_21Assignvc/dense_2/biassave_6/RestoreV2:21*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0*
validate_shape(
╖
save_6/Assign_22Assignvc/dense_2/bias/Adamsave_6/RestoreV2:22*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
╣
save_6/Assign_23Assignvc/dense_2/bias/Adam_1save_6/RestoreV2:23*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0
╗
save_6/Assign_24Assignvc/dense_2/kernelsave_6/RestoreV2:24*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel
└
save_6/Assign_25Assignvc/dense_2/kernel/Adamsave_6/RestoreV2:25*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	А*
validate_shape(
┬
save_6/Assign_26Assignvc/dense_2/kernel/Adam_1save_6/RestoreV2:26*
validate_shape(*
T0*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel*
use_locking(
п
save_6/Assign_27Assignvf/dense/biassave_6/RestoreV2:27*
_output_shapes	
:А* 
_class
loc:@vf/dense/bias*
T0*
validate_shape(*
use_locking(
┤
save_6/Assign_28Assignvf/dense/bias/Adamsave_6/RestoreV2:28*
_output_shapes	
:А*
T0*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias
╢
save_6/Assign_29Assignvf/dense/bias/Adam_1save_6/RestoreV2:29* 
_class
loc:@vf/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:А*
T0
╖
save_6/Assign_30Assignvf/dense/kernelsave_6/RestoreV2:30*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<А*
T0
╝
save_6/Assign_31Assignvf/dense/kernel/Adamsave_6/RestoreV2:31*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	<А*
use_locking(
╛
save_6/Assign_32Assignvf/dense/kernel/Adam_1save_6/RestoreV2:32*
use_locking(*
validate_shape(*
_output_shapes
:	<А*"
_class
loc:@vf/dense/kernel*
T0
│
save_6/Assign_33Assignvf/dense_1/biassave_6/RestoreV2:33*
_output_shapes	
:А*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vf/dense_1/bias
╕
save_6/Assign_34Assignvf/dense_1/bias/Adamsave_6/RestoreV2:34*
_output_shapes	
:А*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vf/dense_1/bias
║
save_6/Assign_35Assignvf/dense_1/bias/Adam_1save_6/RestoreV2:35*
_output_shapes	
:А*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0
╝
save_6/Assign_36Assignvf/dense_1/kernelsave_6/RestoreV2:36*
T0*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:
АА
┴
save_6/Assign_37Assignvf/dense_1/kernel/Adamsave_6/RestoreV2:37*
T0* 
_output_shapes
:
АА*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(
├
save_6/Assign_38Assignvf/dense_1/kernel/Adam_1save_6/RestoreV2:38* 
_output_shapes
:
АА*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(*
validate_shape(
▓
save_6/Assign_39Assignvf/dense_2/biassave_6/RestoreV2:39*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@vf/dense_2/bias
╖
save_6/Assign_40Assignvf/dense_2/bias/Adamsave_6/RestoreV2:40*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias
╣
save_6/Assign_41Assignvf/dense_2/bias/Adam_1save_6/RestoreV2:41*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
╗
save_6/Assign_42Assignvf/dense_2/kernelsave_6/RestoreV2:42*
T0*
validate_shape(*
_output_shapes
:	А*
use_locking(*$
_class
loc:@vf/dense_2/kernel
└
save_6/Assign_43Assignvf/dense_2/kernel/Adamsave_6/RestoreV2:43*
validate_shape(*
_output_shapes
:	А*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
T0
┬
save_6/Assign_44Assignvf/dense_2/kernel/Adam_1save_6/RestoreV2:44*
use_locking(*
T0*
_output_shapes
:	А*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
ч
save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_2^save_6/Assign_20^save_6/Assign_21^save_6/Assign_22^save_6/Assign_23^save_6/Assign_24^save_6/Assign_25^save_6/Assign_26^save_6/Assign_27^save_6/Assign_28^save_6/Assign_29^save_6/Assign_3^save_6/Assign_30^save_6/Assign_31^save_6/Assign_32^save_6/Assign_33^save_6/Assign_34^save_6/Assign_35^save_6/Assign_36^save_6/Assign_37^save_6/Assign_38^save_6/Assign_39^save_6/Assign_4^save_6/Assign_40^save_6/Assign_41^save_6/Assign_42^save_6/Assign_43^save_6/Assign_44^save_6/Assign_5^save_6/Assign_6^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard
[
save_7/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_7/filenamePlaceholderWithDefaultsave_7/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_7/ConstPlaceholderWithDefaultsave_7/filename*
shape: *
dtype0*
_output_shapes
: 
Ж
save_7/StringJoin/inputs_1Const*<
value3B1 B+_temp_3bfddc64066e4983b8f52e455331d3d0/part*
dtype0*
_output_shapes
: 
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_7/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_7/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
Е
save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
Є
save_7/SaveV2/tensor_namesConst*
_output_shapes
:-*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
┐
save_7/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
╢
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-
Щ
save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*)
_class
loc:@save_7/ShardedFilename*
_output_shapes
: *
T0
г
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*

axis *
_output_shapes
:*
T0*
N
Г
save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(
В
save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency*
_output_shapes
: *
T0
ї
save_7/RestoreV2/tensor_namesConst*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
┬
!save_7/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ў
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*;
dtypes1
/2-*╩
_output_shapes╖
┤:::::::::::::::::::::::::::::::::::::::::::::
в
save_7/AssignAssignbeta1_powersave_7/RestoreV2* 
_class
loc:@vc/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
: 
ж
save_7/Assign_1Assignbeta2_powersave_7/RestoreV2:1*
_output_shapes
: *
T0*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias
н
save_7/Assign_2Assignpi/dense/biassave_7/RestoreV2:2*
_output_shapes	
:А*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(*
T0
╡
save_7/Assign_3Assignpi/dense/kernelsave_7/RestoreV2:3*
_output_shapes
:	<А*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(*
T0
▒
save_7/Assign_4Assignpi/dense_1/biassave_7/RestoreV2:4*
_output_shapes	
:А*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias
║
save_7/Assign_5Assignpi/dense_1/kernelsave_7/RestoreV2:5*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(* 
_output_shapes
:
АА
░
save_7/Assign_6Assignpi/dense_2/biassave_7/RestoreV2:6*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:
╣
save_7/Assign_7Assignpi/dense_2/kernelsave_7/RestoreV2:7*
_output_shapes
:	А*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(*
T0
ж
save_7/Assign_8Assign
pi/log_stdsave_7/RestoreV2:8*
validate_shape(*
use_locking(*
_class
loc:@pi/log_std*
_output_shapes
:*
T0
н
save_7/Assign_9Assignvc/dense/biassave_7/RestoreV2:9*
use_locking(*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:А
┤
save_7/Assign_10Assignvc/dense/bias/Adamsave_7/RestoreV2:10*
validate_shape(*
use_locking(*
_output_shapes	
:А*
T0* 
_class
loc:@vc/dense/bias
╢
save_7/Assign_11Assignvc/dense/bias/Adam_1save_7/RestoreV2:11*
T0*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:А
╖
save_7/Assign_12Assignvc/dense/kernelsave_7/RestoreV2:12*
_output_shapes
:	<А*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(
╝
save_7/Assign_13Assignvc/dense/kernel/Adamsave_7/RestoreV2:13*
use_locking(*
T0*
_output_shapes
:	<А*"
_class
loc:@vc/dense/kernel*
validate_shape(
╛
save_7/Assign_14Assignvc/dense/kernel/Adam_1save_7/RestoreV2:14*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<А
│
save_7/Assign_15Assignvc/dense_1/biassave_7/RestoreV2:15*"
_class
loc:@vc/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:А*
T0
╕
save_7/Assign_16Assignvc/dense_1/bias/Adamsave_7/RestoreV2:16*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:А*
validate_shape(
║
save_7/Assign_17Assignvc/dense_1/bias/Adam_1save_7/RestoreV2:17*
use_locking(*
T0*
_output_shapes	
:А*
validate_shape(*"
_class
loc:@vc/dense_1/bias
╝
save_7/Assign_18Assignvc/dense_1/kernelsave_7/RestoreV2:18*
use_locking(* 
_output_shapes
:
АА*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel
┴
save_7/Assign_19Assignvc/dense_1/kernel/Adamsave_7/RestoreV2:19* 
_output_shapes
:
АА*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel
├
save_7/Assign_20Assignvc/dense_1/kernel/Adam_1save_7/RestoreV2:20* 
_output_shapes
:
АА*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
T0*
validate_shape(
▓
save_7/Assign_21Assignvc/dense_2/biassave_7/RestoreV2:21*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias
╖
save_7/Assign_22Assignvc/dense_2/bias/Adamsave_7/RestoreV2:22*
T0*
use_locking(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(
╣
save_7/Assign_23Assignvc/dense_2/bias/Adam_1save_7/RestoreV2:23*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
use_locking(
╗
save_7/Assign_24Assignvc/dense_2/kernelsave_7/RestoreV2:24*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(*
T0
└
save_7/Assign_25Assignvc/dense_2/kernel/Adamsave_7/RestoreV2:25*
validate_shape(*
_output_shapes
:	А*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel
┬
save_7/Assign_26Assignvc/dense_2/kernel/Adam_1save_7/RestoreV2:26*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	А*
T0*
use_locking(
п
save_7/Assign_27Assignvf/dense/biassave_7/RestoreV2:27*
T0*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:А
┤
save_7/Assign_28Assignvf/dense/bias/Adamsave_7/RestoreV2:28* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:А*
validate_shape(*
use_locking(
╢
save_7/Assign_29Assignvf/dense/bias/Adam_1save_7/RestoreV2:29*
_output_shapes	
:А*
use_locking(*
validate_shape(*
T0* 
_class
loc:@vf/dense/bias
╖
save_7/Assign_30Assignvf/dense/kernelsave_7/RestoreV2:30*"
_class
loc:@vf/dense/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	<А
╝
save_7/Assign_31Assignvf/dense/kernel/Adamsave_7/RestoreV2:31*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	<А
╛
save_7/Assign_32Assignvf/dense/kernel/Adam_1save_7/RestoreV2:32*
_output_shapes
:	<А*
T0*
validate_shape(*"
_class
loc:@vf/dense/kernel*
use_locking(
│
save_7/Assign_33Assignvf/dense_1/biassave_7/RestoreV2:33*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:А
╕
save_7/Assign_34Assignvf/dense_1/bias/Adamsave_7/RestoreV2:34*
T0*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:А
║
save_7/Assign_35Assignvf/dense_1/bias/Adam_1save_7/RestoreV2:35*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:А
╝
save_7/Assign_36Assignvf/dense_1/kernelsave_7/RestoreV2:36*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
АА*
T0*
validate_shape(
┴
save_7/Assign_37Assignvf/dense_1/kernel/Adamsave_7/RestoreV2:37*
use_locking(* 
_output_shapes
:
АА*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0
├
save_7/Assign_38Assignvf/dense_1/kernel/Adam_1save_7/RestoreV2:38* 
_output_shapes
:
АА*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(
▓
save_7/Assign_39Assignvf/dense_2/biassave_7/RestoreV2:39*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias*
validate_shape(
╖
save_7/Assign_40Assignvf/dense_2/bias/Adamsave_7/RestoreV2:40*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
╣
save_7/Assign_41Assignvf/dense_2/bias/Adam_1save_7/RestoreV2:41*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
╗
save_7/Assign_42Assignvf/dense_2/kernelsave_7/RestoreV2:42*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	А
└
save_7/Assign_43Assignvf/dense_2/kernel/Adamsave_7/RestoreV2:43*
_output_shapes
:	А*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0
┬
save_7/Assign_44Assignvf/dense_2/kernel/Adam_1save_7/RestoreV2:44*
T0*
_output_shapes
:	А*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel
ч
save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_2^save_7/Assign_20^save_7/Assign_21^save_7/Assign_22^save_7/Assign_23^save_7/Assign_24^save_7/Assign_25^save_7/Assign_26^save_7/Assign_27^save_7/Assign_28^save_7/Assign_29^save_7/Assign_3^save_7/Assign_30^save_7/Assign_31^save_7/Assign_32^save_7/Assign_33^save_7/Assign_34^save_7/Assign_35^save_7/Assign_36^save_7/Assign_37^save_7/Assign_38^save_7/Assign_39^save_7/Assign_4^save_7/Assign_40^save_7/Assign_41^save_7/Assign_42^save_7/Assign_43^save_7/Assign_44^save_7/Assign_5^save_7/Assign_6^save_7/Assign_7^save_7/Assign_8^save_7/Assign_9
1
save_7/restore_allNoOp^save_7/restore_shard
[
save_8/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_8/filenamePlaceholderWithDefaultsave_8/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_8/ConstPlaceholderWithDefaultsave_8/filename*
shape: *
_output_shapes
: *
dtype0
Ж
save_8/StringJoin/inputs_1Const*<
value3B1 B+_temp_b21080f066c447d683e1b8619d68f91c/part*
_output_shapes
: *
dtype0
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_8/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
^
save_8/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
Е
save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
Є
save_8/SaveV2/tensor_namesConst*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
┐
save_8/SaveV2/shape_and_slicesConst*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-
╢
save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-
Щ
save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*
T0*)
_class
loc:@save_8/ShardedFilename*
_output_shapes
: 
г
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilename^save_8/control_dependency*
T0*
_output_shapes
:*
N*

axis 
Г
save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(
В
save_8/IdentityIdentitysave_8/Const^save_8/MergeV2Checkpoints^save_8/control_dependency*
T0*
_output_shapes
: 
ї
save_8/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:-*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
┬
!save_8/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ў
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*╩
_output_shapes╖
┤:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
в
save_8/AssignAssignbeta1_powersave_8/RestoreV2*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes
: 
ж
save_8/Assign_1Assignbeta2_powersave_8/RestoreV2:1* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(*
T0
н
save_8/Assign_2Assignpi/dense/biassave_8/RestoreV2:2*
use_locking(*
_output_shapes	
:А*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
╡
save_8/Assign_3Assignpi/dense/kernelsave_8/RestoreV2:3*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<А*
T0*
validate_shape(*
use_locking(
▒
save_8/Assign_4Assignpi/dense_1/biassave_8/RestoreV2:4*
T0*
_output_shapes	
:А*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(
║
save_8/Assign_5Assignpi/dense_1/kernelsave_8/RestoreV2:5* 
_output_shapes
:
АА*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
░
save_8/Assign_6Assignpi/dense_2/biassave_8/RestoreV2:6*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
╣
save_8/Assign_7Assignpi/dense_2/kernelsave_8/RestoreV2:7*
T0*
_output_shapes
:	А*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
ж
save_8/Assign_8Assign
pi/log_stdsave_8/RestoreV2:8*
use_locking(*
validate_shape(*
T0*
_class
loc:@pi/log_std*
_output_shapes
:
н
save_8/Assign_9Assignvc/dense/biassave_8/RestoreV2:9*
T0*
_output_shapes	
:А* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(
┤
save_8/Assign_10Assignvc/dense/bias/Adamsave_8/RestoreV2:10* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
_output_shapes	
:А*
use_locking(
╢
save_8/Assign_11Assignvc/dense/bias/Adam_1save_8/RestoreV2:11*
_output_shapes	
:А*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vc/dense/bias
╖
save_8/Assign_12Assignvc/dense/kernelsave_8/RestoreV2:12*
use_locking(*
T0*
_output_shapes
:	<А*
validate_shape(*"
_class
loc:@vc/dense/kernel
╝
save_8/Assign_13Assignvc/dense/kernel/Adamsave_8/RestoreV2:13*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	<А*"
_class
loc:@vc/dense/kernel
╛
save_8/Assign_14Assignvc/dense/kernel/Adam_1save_8/RestoreV2:14*
T0*
validate_shape(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<А*
use_locking(
│
save_8/Assign_15Assignvc/dense_1/biassave_8/RestoreV2:15*
use_locking(*
T0*
_output_shapes	
:А*
validate_shape(*"
_class
loc:@vc/dense_1/bias
╕
save_8/Assign_16Assignvc/dense_1/bias/Adamsave_8/RestoreV2:16*
_output_shapes	
:А*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
validate_shape(
║
save_8/Assign_17Assignvc/dense_1/bias/Adam_1save_8/RestoreV2:17*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:А*
T0*
validate_shape(*
use_locking(
╝
save_8/Assign_18Assignvc/dense_1/kernelsave_8/RestoreV2:18*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
АА
┴
save_8/Assign_19Assignvc/dense_1/kernel/Adamsave_8/RestoreV2:19*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel
├
save_8/Assign_20Assignvc/dense_1/kernel/Adam_1save_8/RestoreV2:20*
validate_shape(*
use_locking(* 
_output_shapes
:
АА*$
_class
loc:@vc/dense_1/kernel*
T0
▓
save_8/Assign_21Assignvc/dense_2/biassave_8/RestoreV2:21*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(
╖
save_8/Assign_22Assignvc/dense_2/bias/Adamsave_8/RestoreV2:22*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
╣
save_8/Assign_23Assignvc/dense_2/bias/Adam_1save_8/RestoreV2:23*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(
╗
save_8/Assign_24Assignvc/dense_2/kernelsave_8/RestoreV2:24*
_output_shapes
:	А*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
└
save_8/Assign_25Assignvc/dense_2/kernel/Adamsave_8/RestoreV2:25*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel
┬
save_8/Assign_26Assignvc/dense_2/kernel/Adam_1save_8/RestoreV2:26*
_output_shapes
:	А*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel
п
save_8/Assign_27Assignvf/dense/biassave_8/RestoreV2:27*
T0* 
_class
loc:@vf/dense/bias*
use_locking(*
_output_shapes	
:А*
validate_shape(
┤
save_8/Assign_28Assignvf/dense/bias/Adamsave_8/RestoreV2:28* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:А
╢
save_8/Assign_29Assignvf/dense/bias/Adam_1save_8/RestoreV2:29* 
_class
loc:@vf/dense/bias*
_output_shapes	
:А*
validate_shape(*
T0*
use_locking(
╖
save_8/Assign_30Assignvf/dense/kernelsave_8/RestoreV2:30*
_output_shapes
:	<А*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense/kernel
╝
save_8/Assign_31Assignvf/dense/kernel/Adamsave_8/RestoreV2:31*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<А*
use_locking(*
validate_shape(*
T0
╛
save_8/Assign_32Assignvf/dense/kernel/Adam_1save_8/RestoreV2:32*
_output_shapes
:	<А*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel
│
save_8/Assign_33Assignvf/dense_1/biassave_8/RestoreV2:33*
validate_shape(*
_output_shapes	
:А*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(
╕
save_8/Assign_34Assignvf/dense_1/bias/Adamsave_8/RestoreV2:34*
use_locking(*
validate_shape(*
_output_shapes	
:А*"
_class
loc:@vf/dense_1/bias*
T0
║
save_8/Assign_35Assignvf/dense_1/bias/Adam_1save_8/RestoreV2:35*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:А
╝
save_8/Assign_36Assignvf/dense_1/kernelsave_8/RestoreV2:36* 
_output_shapes
:
АА*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0
┴
save_8/Assign_37Assignvf/dense_1/kernel/Adamsave_8/RestoreV2:37* 
_output_shapes
:
АА*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(*
T0
├
save_8/Assign_38Assignvf/dense_1/kernel/Adam_1save_8/RestoreV2:38* 
_output_shapes
:
АА*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
T0
▓
save_8/Assign_39Assignvf/dense_2/biassave_8/RestoreV2:39*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
╖
save_8/Assign_40Assignvf/dense_2/bias/Adamsave_8/RestoreV2:40*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
╣
save_8/Assign_41Assignvf/dense_2/bias/Adam_1save_8/RestoreV2:41*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:
╗
save_8/Assign_42Assignvf/dense_2/kernelsave_8/RestoreV2:42*
T0*
_output_shapes
:	А*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel
└
save_8/Assign_43Assignvf/dense_2/kernel/Adamsave_8/RestoreV2:43*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	А
┬
save_8/Assign_44Assignvf/dense_2/kernel/Adam_1save_8/RestoreV2:44*
use_locking(*
_output_shapes
:	А*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0
ч
save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_16^save_8/Assign_17^save_8/Assign_18^save_8/Assign_19^save_8/Assign_2^save_8/Assign_20^save_8/Assign_21^save_8/Assign_22^save_8/Assign_23^save_8/Assign_24^save_8/Assign_25^save_8/Assign_26^save_8/Assign_27^save_8/Assign_28^save_8/Assign_29^save_8/Assign_3^save_8/Assign_30^save_8/Assign_31^save_8/Assign_32^save_8/Assign_33^save_8/Assign_34^save_8/Assign_35^save_8/Assign_36^save_8/Assign_37^save_8/Assign_38^save_8/Assign_39^save_8/Assign_4^save_8/Assign_40^save_8/Assign_41^save_8/Assign_42^save_8/Assign_43^save_8/Assign_44^save_8/Assign_5^save_8/Assign_6^save_8/Assign_7^save_8/Assign_8^save_8/Assign_9
1
save_8/restore_allNoOp^save_8/restore_shard
[
save_9/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_9/filenamePlaceholderWithDefaultsave_9/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_9/ConstPlaceholderWithDefaultsave_9/filename*
dtype0*
shape: *
_output_shapes
: 
Ж
save_9/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_acb4b8ec71f345528fd472affa260b47/part*
dtype0
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_9/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_9/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
Е
save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
Є
save_9/SaveV2/tensor_namesConst*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
┐
save_9/SaveV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
╢
save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-
Щ
save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*)
_class
loc:@save_9/ShardedFilename*
T0*
_output_shapes
: 
г
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilename^save_9/control_dependency*
N*

axis *
_output_shapes
:*
T0
Г
save_9/MergeV2CheckpointsMergeV2Checkpoints-save_9/MergeV2Checkpoints/checkpoint_prefixessave_9/Const*
delete_old_dirs(
В
save_9/IdentityIdentitysave_9/Const^save_9/MergeV2Checkpoints^save_9/control_dependency*
T0*
_output_shapes
: 
ї
save_9/RestoreV2/tensor_namesConst*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
┬
!save_9/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ў
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*;
dtypes1
/2-*╩
_output_shapes╖
┤:::::::::::::::::::::::::::::::::::::::::::::
в
save_9/AssignAssignbeta1_powersave_9/RestoreV2*
use_locking(*
T0*
_output_shapes
: *
validate_shape(* 
_class
loc:@vc/dense/bias
ж
save_9/Assign_1Assignbeta2_powersave_9/RestoreV2:1*
_output_shapes
: *
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(
н
save_9/Assign_2Assignpi/dense/biassave_9/RestoreV2:2* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
_output_shapes	
:А*
use_locking(
╡
save_9/Assign_3Assignpi/dense/kernelsave_9/RestoreV2:3*
_output_shapes
:	<А*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel
▒
save_9/Assign_4Assignpi/dense_1/biassave_9/RestoreV2:4*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:А
║
save_9/Assign_5Assignpi/dense_1/kernelsave_9/RestoreV2:5* 
_output_shapes
:
АА*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
░
save_9/Assign_6Assignpi/dense_2/biassave_9/RestoreV2:6*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(
╣
save_9/Assign_7Assignpi/dense_2/kernelsave_9/RestoreV2:7*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	А*
validate_shape(
ж
save_9/Assign_8Assign
pi/log_stdsave_9/RestoreV2:8*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*
_class
loc:@pi/log_std
н
save_9/Assign_9Assignvc/dense/biassave_9/RestoreV2:9*
_output_shapes	
:А* 
_class
loc:@vc/dense/bias*
use_locking(*
T0*
validate_shape(
┤
save_9/Assign_10Assignvc/dense/bias/Adamsave_9/RestoreV2:10*
validate_shape(*
_output_shapes	
:А* 
_class
loc:@vc/dense/bias*
use_locking(*
T0
╢
save_9/Assign_11Assignvc/dense/bias/Adam_1save_9/RestoreV2:11*
validate_shape(*
T0*
_output_shapes	
:А*
use_locking(* 
_class
loc:@vc/dense/bias
╖
save_9/Assign_12Assignvc/dense/kernelsave_9/RestoreV2:12*
use_locking(*
validate_shape(*
_output_shapes
:	<А*
T0*"
_class
loc:@vc/dense/kernel
╝
save_9/Assign_13Assignvc/dense/kernel/Adamsave_9/RestoreV2:13*
_output_shapes
:	<А*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vc/dense/kernel
╛
save_9/Assign_14Assignvc/dense/kernel/Adam_1save_9/RestoreV2:14*
use_locking(*
_output_shapes
:	<А*
T0*
validate_shape(*"
_class
loc:@vc/dense/kernel
│
save_9/Assign_15Assignvc/dense_1/biassave_9/RestoreV2:15*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(*
_output_shapes	
:А
╕
save_9/Assign_16Assignvc/dense_1/bias/Adamsave_9/RestoreV2:16*
use_locking(*
_output_shapes	
:А*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
T0
║
save_9/Assign_17Assignvc/dense_1/bias/Adam_1save_9/RestoreV2:17*
_output_shapes	
:А*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(
╝
save_9/Assign_18Assignvc/dense_1/kernelsave_9/RestoreV2:18*
T0* 
_output_shapes
:
АА*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
┴
save_9/Assign_19Assignvc/dense_1/kernel/Adamsave_9/RestoreV2:19*
T0* 
_output_shapes
:
АА*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
use_locking(
├
save_9/Assign_20Assignvc/dense_1/kernel/Adam_1save_9/RestoreV2:20*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
АА
▓
save_9/Assign_21Assignvc/dense_2/biassave_9/RestoreV2:21*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@vc/dense_2/bias
╖
save_9/Assign_22Assignvc/dense_2/bias/Adamsave_9/RestoreV2:22*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(*
validate_shape(
╣
save_9/Assign_23Assignvc/dense_2/bias/Adam_1save_9/RestoreV2:23*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias
╗
save_9/Assign_24Assignvc/dense_2/kernelsave_9/RestoreV2:24*
T0*
_output_shapes
:	А*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel
└
save_9/Assign_25Assignvc/dense_2/kernel/Adamsave_9/RestoreV2:25*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel*
T0*
use_locking(*
validate_shape(
┬
save_9/Assign_26Assignvc/dense_2/kernel/Adam_1save_9/RestoreV2:26*
T0*
validate_shape(*
_output_shapes
:	А*$
_class
loc:@vc/dense_2/kernel*
use_locking(
п
save_9/Assign_27Assignvf/dense/biassave_9/RestoreV2:27* 
_class
loc:@vf/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:А
┤
save_9/Assign_28Assignvf/dense/bias/Adamsave_9/RestoreV2:28*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:А
╢
save_9/Assign_29Assignvf/dense/bias/Adam_1save_9/RestoreV2:29*
use_locking(*
T0*
_output_shapes	
:А*
validate_shape(* 
_class
loc:@vf/dense/bias
╖
save_9/Assign_30Assignvf/dense/kernelsave_9/RestoreV2:30*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	<А*
T0
╝
save_9/Assign_31Assignvf/dense/kernel/Adamsave_9/RestoreV2:31*
_output_shapes
:	<А*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(
╛
save_9/Assign_32Assignvf/dense/kernel/Adam_1save_9/RestoreV2:32*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	<А
│
save_9/Assign_33Assignvf/dense_1/biassave_9/RestoreV2:33*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(*
_output_shapes	
:А*
validate_shape(
╕
save_9/Assign_34Assignvf/dense_1/bias/Adamsave_9/RestoreV2:34*
validate_shape(*
use_locking(*
_output_shapes	
:А*
T0*"
_class
loc:@vf/dense_1/bias
║
save_9/Assign_35Assignvf/dense_1/bias/Adam_1save_9/RestoreV2:35*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(*
_output_shapes	
:А
╝
save_9/Assign_36Assignvf/dense_1/kernelsave_9/RestoreV2:36* 
_output_shapes
:
АА*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel
┴
save_9/Assign_37Assignvf/dense_1/kernel/Adamsave_9/RestoreV2:37* 
_output_shapes
:
АА*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(
├
save_9/Assign_38Assignvf/dense_1/kernel/Adam_1save_9/RestoreV2:38*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
АА
▓
save_9/Assign_39Assignvf/dense_2/biassave_9/RestoreV2:39*"
_class
loc:@vf/dense_2/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
╖
save_9/Assign_40Assignvf/dense_2/bias/Adamsave_9/RestoreV2:40*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(
╣
save_9/Assign_41Assignvf/dense_2/bias/Adam_1save_9/RestoreV2:41*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
╗
save_9/Assign_42Assignvf/dense_2/kernelsave_9/RestoreV2:42*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	А*
T0*
validate_shape(*
use_locking(
└
save_9/Assign_43Assignvf/dense_2/kernel/Adamsave_9/RestoreV2:43*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	А
┬
save_9/Assign_44Assignvf/dense_2/kernel/Adam_1save_9/RestoreV2:44*
_output_shapes
:	А*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(
ч
save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_16^save_9/Assign_17^save_9/Assign_18^save_9/Assign_19^save_9/Assign_2^save_9/Assign_20^save_9/Assign_21^save_9/Assign_22^save_9/Assign_23^save_9/Assign_24^save_9/Assign_25^save_9/Assign_26^save_9/Assign_27^save_9/Assign_28^save_9/Assign_29^save_9/Assign_3^save_9/Assign_30^save_9/Assign_31^save_9/Assign_32^save_9/Assign_33^save_9/Assign_34^save_9/Assign_35^save_9/Assign_36^save_9/Assign_37^save_9/Assign_38^save_9/Assign_39^save_9/Assign_4^save_9/Assign_40^save_9/Assign_41^save_9/Assign_42^save_9/Assign_43^save_9/Assign_44^save_9/Assign_5^save_9/Assign_6^save_9/Assign_7^save_9/Assign_8^save_9/Assign_9
1
save_9/restore_allNoOp^save_9/restore_shard
\
save_10/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_10/filenamePlaceholderWithDefaultsave_10/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_10/ConstPlaceholderWithDefaultsave_10/filename*
shape: *
_output_shapes
: *
dtype0
З
save_10/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_4599784e5e844f53ad77fa0690678c83/part*
_output_shapes
: 
~
save_10/StringJoin
StringJoinsave_10/Constsave_10/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_10/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_10/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
Й
save_10/ShardedFilenameShardedFilenamesave_10/StringJoinsave_10/ShardedFilename/shardsave_10/num_shards*
_output_shapes
: 
є
save_10/SaveV2/tensor_namesConst*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
└
save_10/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
║
save_10/SaveV2SaveV2save_10/ShardedFilenamesave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-
Э
save_10/control_dependencyIdentitysave_10/ShardedFilename^save_10/SaveV2**
_class 
loc:@save_10/ShardedFilename*
T0*
_output_shapes
: 
ж
.save_10/MergeV2Checkpoints/checkpoint_prefixesPacksave_10/ShardedFilename^save_10/control_dependency*
_output_shapes
:*
N*

axis *
T0
Ж
save_10/MergeV2CheckpointsMergeV2Checkpoints.save_10/MergeV2Checkpoints/checkpoint_prefixessave_10/Const*
delete_old_dirs(
Ж
save_10/IdentityIdentitysave_10/Const^save_10/MergeV2Checkpoints^save_10/control_dependency*
_output_shapes
: *
T0
Ў
save_10/RestoreV2/tensor_namesConst*г
valueЩBЦ-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
├
"save_10/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
√
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices*╩
_output_shapes╖
┤:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
д
save_10/AssignAssignbeta1_powersave_10/RestoreV2* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
и
save_10/Assign_1Assignbeta2_powersave_10/RestoreV2:1*
T0*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(
п
save_10/Assign_2Assignpi/dense/biassave_10/RestoreV2:2* 
_class
loc:@pi/dense/bias*
_output_shapes	
:А*
T0*
validate_shape(*
use_locking(
╖
save_10/Assign_3Assignpi/dense/kernelsave_10/RestoreV2:3*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	<А*
validate_shape(*
T0
│
save_10/Assign_4Assignpi/dense_1/biassave_10/RestoreV2:4*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:А*"
_class
loc:@pi/dense_1/bias
╝
save_10/Assign_5Assignpi/dense_1/kernelsave_10/RestoreV2:5* 
_output_shapes
:
АА*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel
▓
save_10/Assign_6Assignpi/dense_2/biassave_10/RestoreV2:6*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
╗
save_10/Assign_7Assignpi/dense_2/kernelsave_10/RestoreV2:7*
_output_shapes
:	А*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(
и
save_10/Assign_8Assign
pi/log_stdsave_10/RestoreV2:8*
T0*
_class
loc:@pi/log_std*
use_locking(*
validate_shape(*
_output_shapes
:
п
save_10/Assign_9Assignvc/dense/biassave_10/RestoreV2:9*
use_locking(*
_output_shapes	
:А*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias
╢
save_10/Assign_10Assignvc/dense/bias/Adamsave_10/RestoreV2:10*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:А*
T0
╕
save_10/Assign_11Assignvc/dense/bias/Adam_1save_10/RestoreV2:11*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
_output_shapes	
:А
╣
save_10/Assign_12Assignvc/dense/kernelsave_10/RestoreV2:12*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	<А*
T0
╛
save_10/Assign_13Assignvc/dense/kernel/Adamsave_10/RestoreV2:13*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	<А*
use_locking(*
T0
└
save_10/Assign_14Assignvc/dense/kernel/Adam_1save_10/RestoreV2:14*
use_locking(*
_output_shapes
:	<А*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(
╡
save_10/Assign_15Assignvc/dense_1/biassave_10/RestoreV2:15*
_output_shapes	
:А*"
_class
loc:@vc/dense_1/bias*
T0*
validate_shape(*
use_locking(
║
save_10/Assign_16Assignvc/dense_1/bias/Adamsave_10/RestoreV2:16*
use_locking(*
validate_shape(*
_output_shapes	
:А*"
_class
loc:@vc/dense_1/bias*
T0
╝
save_10/Assign_17Assignvc/dense_1/bias/Adam_1save_10/RestoreV2:17*
T0*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:А
╛
save_10/Assign_18Assignvc/dense_1/kernelsave_10/RestoreV2:18*
T0* 
_output_shapes
:
АА*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
├
save_10/Assign_19Assignvc/dense_1/kernel/Adamsave_10/RestoreV2:19*
use_locking(* 
_output_shapes
:
АА*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
┼
save_10/Assign_20Assignvc/dense_1/kernel/Adam_1save_10/RestoreV2:20*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
АА*$
_class
loc:@vc/dense_1/kernel
┤
save_10/Assign_21Assignvc/dense_2/biassave_10/RestoreV2:21*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
╣
save_10/Assign_22Assignvc/dense_2/bias/Adamsave_10/RestoreV2:22*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
╗
save_10/Assign_23Assignvc/dense_2/bias/Adam_1save_10/RestoreV2:23*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0
╜
save_10/Assign_24Assignvc/dense_2/kernelsave_10/RestoreV2:24*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	А
┬
save_10/Assign_25Assignvc/dense_2/kernel/Adamsave_10/RestoreV2:25*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	А
─
save_10/Assign_26Assignvc/dense_2/kernel/Adam_1save_10/RestoreV2:26*$
_class
loc:@vc/dense_2/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	А
▒
save_10/Assign_27Assignvf/dense/biassave_10/RestoreV2:27*
use_locking(*
_output_shapes	
:А*
validate_shape(* 
_class
loc:@vf/dense/bias*
T0
╢
save_10/Assign_28Assignvf/dense/bias/Adamsave_10/RestoreV2:28* 
_class
loc:@vf/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:А*
T0
╕
save_10/Assign_29Assignvf/dense/bias/Adam_1save_10/RestoreV2:29*
validate_shape(*
use_locking(*
_output_shapes	
:А*
T0* 
_class
loc:@vf/dense/bias
╣
save_10/Assign_30Assignvf/dense/kernelsave_10/RestoreV2:30*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<А*
T0*
use_locking(*
validate_shape(
╛
save_10/Assign_31Assignvf/dense/kernel/Adamsave_10/RestoreV2:31*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<А*
use_locking(*
validate_shape(*
T0
└
save_10/Assign_32Assignvf/dense/kernel/Adam_1save_10/RestoreV2:32*"
_class
loc:@vf/dense/kernel*
use_locking(*
T0*
_output_shapes
:	<А*
validate_shape(
╡
save_10/Assign_33Assignvf/dense_1/biassave_10/RestoreV2:33*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:А
║
save_10/Assign_34Assignvf/dense_1/bias/Adamsave_10/RestoreV2:34*
_output_shapes	
:А*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
use_locking(
╝
save_10/Assign_35Assignvf/dense_1/bias/Adam_1save_10/RestoreV2:35*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:А
╛
save_10/Assign_36Assignvf/dense_1/kernelsave_10/RestoreV2:36*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
АА*
T0*
use_locking(
├
save_10/Assign_37Assignvf/dense_1/kernel/Adamsave_10/RestoreV2:37*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
┼
save_10/Assign_38Assignvf/dense_1/kernel/Adam_1save_10/RestoreV2:38* 
_output_shapes
:
АА*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(*
T0
┤
save_10/Assign_39Assignvf/dense_2/biassave_10/RestoreV2:39*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0
╣
save_10/Assign_40Assignvf/dense_2/bias/Adamsave_10/RestoreV2:40*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0
╗
save_10/Assign_41Assignvf/dense_2/bias/Adam_1save_10/RestoreV2:41*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0
╜
save_10/Assign_42Assignvf/dense_2/kernelsave_10/RestoreV2:42*
T0*
use_locking(*
_output_shapes
:	А*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
┬
save_10/Assign_43Assignvf/dense_2/kernel/Adamsave_10/RestoreV2:43*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	А*
T0*
validate_shape(*
use_locking(
─
save_10/Assign_44Assignvf/dense_2/kernel/Adam_1save_10/RestoreV2:44*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	А
Х
save_10/restore_shardNoOp^save_10/Assign^save_10/Assign_1^save_10/Assign_10^save_10/Assign_11^save_10/Assign_12^save_10/Assign_13^save_10/Assign_14^save_10/Assign_15^save_10/Assign_16^save_10/Assign_17^save_10/Assign_18^save_10/Assign_19^save_10/Assign_2^save_10/Assign_20^save_10/Assign_21^save_10/Assign_22^save_10/Assign_23^save_10/Assign_24^save_10/Assign_25^save_10/Assign_26^save_10/Assign_27^save_10/Assign_28^save_10/Assign_29^save_10/Assign_3^save_10/Assign_30^save_10/Assign_31^save_10/Assign_32^save_10/Assign_33^save_10/Assign_34^save_10/Assign_35^save_10/Assign_36^save_10/Assign_37^save_10/Assign_38^save_10/Assign_39^save_10/Assign_4^save_10/Assign_40^save_10/Assign_41^save_10/Assign_42^save_10/Assign_43^save_10/Assign_44^save_10/Assign_5^save_10/Assign_6^save_10/Assign_7^save_10/Assign_8^save_10/Assign_9
3
save_10/restore_allNoOp^save_10/restore_shard "ЖE
save_10/Const:0save_10/Identity:0save_10/restore_all (5 @F8"
train_op

Adam"Ё*
	variablesт*▀*
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
R
pi/log_std:0pi/log_std/Assignpi/log_std/read:02pi/log_std/initial_value:08
s
vf/dense/kernel:0vf/dense/kernel/Assignvf/dense/kernel/read:02,vf/dense/kernel/Initializer/random_uniform:08
b
vf/dense/bias:0vf/dense/bias/Assignvf/dense/bias/read:02!vf/dense/bias/Initializer/zeros:08
{
vf/dense_1/kernel:0vf/dense_1/kernel/Assignvf/dense_1/kernel/read:02.vf/dense_1/kernel/Initializer/random_uniform:08
j
vf/dense_1/bias:0vf/dense_1/bias/Assignvf/dense_1/bias/read:02#vf/dense_1/bias/Initializer/zeros:08
{
vf/dense_2/kernel:0vf/dense_2/kernel/Assignvf/dense_2/kernel/read:02.vf/dense_2/kernel/Initializer/random_uniform:08
j
vf/dense_2/bias:0vf/dense_2/bias/Assignvf/dense_2/bias/read:02#vf/dense_2/bias/Initializer/zeros:08
s
vc/dense/kernel:0vc/dense/kernel/Assignvc/dense/kernel/read:02,vc/dense/kernel/Initializer/random_uniform:08
b
vc/dense/bias:0vc/dense/bias/Assignvc/dense/bias/read:02!vc/dense/bias/Initializer/zeros:08
{
vc/dense_1/kernel:0vc/dense_1/kernel/Assignvc/dense_1/kernel/read:02.vc/dense_1/kernel/Initializer/random_uniform:08
j
vc/dense_1/bias:0vc/dense_1/bias/Assignvc/dense_1/bias/read:02#vc/dense_1/bias/Initializer/zeros:08
{
vc/dense_2/kernel:0vc/dense_2/kernel/Assignvc/dense_2/kernel/read:02.vc/dense_2/kernel/Initializer/random_uniform:08
j
vc/dense_2/bias:0vc/dense_2/bias/Assignvc/dense_2/bias/read:02#vc/dense_2/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
|
vf/dense/kernel/Adam:0vf/dense/kernel/Adam/Assignvf/dense/kernel/Adam/read:02(vf/dense/kernel/Adam/Initializer/zeros:0
Д
vf/dense/kernel/Adam_1:0vf/dense/kernel/Adam_1/Assignvf/dense/kernel/Adam_1/read:02*vf/dense/kernel/Adam_1/Initializer/zeros:0
t
vf/dense/bias/Adam:0vf/dense/bias/Adam/Assignvf/dense/bias/Adam/read:02&vf/dense/bias/Adam/Initializer/zeros:0
|
vf/dense/bias/Adam_1:0vf/dense/bias/Adam_1/Assignvf/dense/bias/Adam_1/read:02(vf/dense/bias/Adam_1/Initializer/zeros:0
Д
vf/dense_1/kernel/Adam:0vf/dense_1/kernel/Adam/Assignvf/dense_1/kernel/Adam/read:02*vf/dense_1/kernel/Adam/Initializer/zeros:0
М
vf/dense_1/kernel/Adam_1:0vf/dense_1/kernel/Adam_1/Assignvf/dense_1/kernel/Adam_1/read:02,vf/dense_1/kernel/Adam_1/Initializer/zeros:0
|
vf/dense_1/bias/Adam:0vf/dense_1/bias/Adam/Assignvf/dense_1/bias/Adam/read:02(vf/dense_1/bias/Adam/Initializer/zeros:0
Д
vf/dense_1/bias/Adam_1:0vf/dense_1/bias/Adam_1/Assignvf/dense_1/bias/Adam_1/read:02*vf/dense_1/bias/Adam_1/Initializer/zeros:0
Д
vf/dense_2/kernel/Adam:0vf/dense_2/kernel/Adam/Assignvf/dense_2/kernel/Adam/read:02*vf/dense_2/kernel/Adam/Initializer/zeros:0
М
vf/dense_2/kernel/Adam_1:0vf/dense_2/kernel/Adam_1/Assignvf/dense_2/kernel/Adam_1/read:02,vf/dense_2/kernel/Adam_1/Initializer/zeros:0
|
vf/dense_2/bias/Adam:0vf/dense_2/bias/Adam/Assignvf/dense_2/bias/Adam/read:02(vf/dense_2/bias/Adam/Initializer/zeros:0
Д
vf/dense_2/bias/Adam_1:0vf/dense_2/bias/Adam_1/Assignvf/dense_2/bias/Adam_1/read:02*vf/dense_2/bias/Adam_1/Initializer/zeros:0
|
vc/dense/kernel/Adam:0vc/dense/kernel/Adam/Assignvc/dense/kernel/Adam/read:02(vc/dense/kernel/Adam/Initializer/zeros:0
Д
vc/dense/kernel/Adam_1:0vc/dense/kernel/Adam_1/Assignvc/dense/kernel/Adam_1/read:02*vc/dense/kernel/Adam_1/Initializer/zeros:0
t
vc/dense/bias/Adam:0vc/dense/bias/Adam/Assignvc/dense/bias/Adam/read:02&vc/dense/bias/Adam/Initializer/zeros:0
|
vc/dense/bias/Adam_1:0vc/dense/bias/Adam_1/Assignvc/dense/bias/Adam_1/read:02(vc/dense/bias/Adam_1/Initializer/zeros:0
Д
vc/dense_1/kernel/Adam:0vc/dense_1/kernel/Adam/Assignvc/dense_1/kernel/Adam/read:02*vc/dense_1/kernel/Adam/Initializer/zeros:0
М
vc/dense_1/kernel/Adam_1:0vc/dense_1/kernel/Adam_1/Assignvc/dense_1/kernel/Adam_1/read:02,vc/dense_1/kernel/Adam_1/Initializer/zeros:0
|
vc/dense_1/bias/Adam:0vc/dense_1/bias/Adam/Assignvc/dense_1/bias/Adam/read:02(vc/dense_1/bias/Adam/Initializer/zeros:0
Д
vc/dense_1/bias/Adam_1:0vc/dense_1/bias/Adam_1/Assignvc/dense_1/bias/Adam_1/read:02*vc/dense_1/bias/Adam_1/Initializer/zeros:0
Д
vc/dense_2/kernel/Adam:0vc/dense_2/kernel/Adam/Assignvc/dense_2/kernel/Adam/read:02*vc/dense_2/kernel/Adam/Initializer/zeros:0
М
vc/dense_2/kernel/Adam_1:0vc/dense_2/kernel/Adam_1/Assignvc/dense_2/kernel/Adam_1/read:02,vc/dense_2/kernel/Adam_1/Initializer/zeros:0
|
vc/dense_2/bias/Adam:0vc/dense_2/bias/Adam/Assignvc/dense_2/bias/Adam/read:02(vc/dense_2/bias/Adam/Initializer/zeros:0
Д
vc/dense_2/bias/Adam_1:0vc/dense_2/bias/Adam_1/Assignvc/dense_2/bias/Adam_1/read:02*vc/dense_2/bias/Adam_1/Initializer/zeros:0"Ё
trainable_variables╪╒
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
R
pi/log_std:0pi/log_std/Assignpi/log_std/read:02pi/log_std/initial_value:08
s
vf/dense/kernel:0vf/dense/kernel/Assignvf/dense/kernel/read:02,vf/dense/kernel/Initializer/random_uniform:08
b
vf/dense/bias:0vf/dense/bias/Assignvf/dense/bias/read:02!vf/dense/bias/Initializer/zeros:08
{
vf/dense_1/kernel:0vf/dense_1/kernel/Assignvf/dense_1/kernel/read:02.vf/dense_1/kernel/Initializer/random_uniform:08
j
vf/dense_1/bias:0vf/dense_1/bias/Assignvf/dense_1/bias/read:02#vf/dense_1/bias/Initializer/zeros:08
{
vf/dense_2/kernel:0vf/dense_2/kernel/Assignvf/dense_2/kernel/read:02.vf/dense_2/kernel/Initializer/random_uniform:08
j
vf/dense_2/bias:0vf/dense_2/bias/Assignvf/dense_2/bias/read:02#vf/dense_2/bias/Initializer/zeros:08
s
vc/dense/kernel:0vc/dense/kernel/Assignvc/dense/kernel/read:02,vc/dense/kernel/Initializer/random_uniform:08
b
vc/dense/bias:0vc/dense/bias/Assignvc/dense/bias/read:02!vc/dense/bias/Initializer/zeros:08
{
vc/dense_1/kernel:0vc/dense_1/kernel/Assignvc/dense_1/kernel/read:02.vc/dense_1/kernel/Initializer/random_uniform:08
j
vc/dense_1/bias:0vc/dense_1/bias/Assignvc/dense_1/bias/read:02#vc/dense_1/bias/Initializer/zeros:08
{
vc/dense_2/kernel:0vc/dense_2/kernel/Assignvc/dense_2/kernel/read:02.vc/dense_2/kernel/Initializer/random_uniform:08
j
vc/dense_2/bias:0vc/dense_2/bias/Assignvc/dense_2/bias/read:02#vc/dense_2/bias/Initializer/zeros:08*╧
serving_default╗
)
x$
Placeholder:0         <%
vc
vc/Squeeze:0         $
v
vf/Squeeze:0         %
pi
pi/add:0         tensorflow/serving/predict