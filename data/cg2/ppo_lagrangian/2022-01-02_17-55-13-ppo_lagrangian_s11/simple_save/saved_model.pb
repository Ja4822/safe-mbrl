ů
˙%Ř%
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
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
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

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
delete_old_dirsbool(
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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

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
list(type)(

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
0
Sigmoid
x"T
y"T"
Ttype:

2
@
Softplus
features"T
activations"T"
Ttype:
2

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

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
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.15.42v1.15.3-68-gdf8c55c×
n
PlaceholderPlaceholder*
shape:˙˙˙˙˙˙˙˙˙<*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<
p
Placeholder_1Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_2Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
Placeholder_3Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
Placeholder_4Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
Placeholder_5Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_6Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
N
Placeholder_7Placeholder*
dtype0*
shape: *
_output_shapes
: 
N
Placeholder_8Placeholder*
shape: *
_output_shapes
: *
dtype0
Ľ
0pi/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@pi/dense/kernel*
valueB"<      

.pi/dense/kernel/Initializer/random_uniform/minConst*"
_class
loc:@pi/dense/kernel*
dtype0*
valueB
 *ž*
_output_shapes
: 

.pi/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *>*"
_class
loc:@pi/dense/kernel*
dtype0*
_output_shapes
: 
ď
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@pi/dense/kernel*
dtype0*

seed*
_output_shapes
:	<*
seed2
Ú
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *"
_class
loc:@pi/dense/kernel*
T0
í
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel
ß
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<
Š
pi/dense/kernel
VariableV2*
_output_shapes
:	<*
	container *
shape:	<*"
_class
loc:@pi/dense/kernel*
shared_name *
dtype0
Ô
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*
T0*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(

pi/dense/kernel/readIdentitypi/dense/kernel*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<

pi/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    * 
_class
loc:@pi/dense/bias

pi/dense/bias
VariableV2*
	container *
dtype0* 
_class
loc:@pi/dense/bias*
shape:*
_output_shapes	
:*
shared_name 
ż
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
use_locking(
u
pi/dense/bias/readIdentitypi/dense/bias*
T0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias

pi/dense/MatMulMatMulPlaceholderpi/dense/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( 

pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Z
pi/dense/TanhTanhpi/dense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
2pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*$
_class
loc:@pi/dense_1/kernel*
valueB"      *
_output_shapes
:

0pi/dense_1/kernel/Initializer/random_uniform/minConst*$
_class
loc:@pi/dense_1/kernel*
valueB
 *×łÝ˝*
_output_shapes
: *
dtype0

0pi/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *×łÝ=*$
_class
loc:@pi/dense_1/kernel*
_output_shapes
: 
ö
:pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_1/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
seed2*
T0*

seed
â
0pi/dense_1/kernel/Initializer/random_uniform/subSub0pi/dense_1/kernel/Initializer/random_uniform/max0pi/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@pi/dense_1/kernel
ö
0pi/dense_1/kernel/Initializer/random_uniform/mulMul:pi/dense_1/kernel/Initializer/random_uniform/RandomUniform0pi/dense_1/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel
č
,pi/dense_1/kernel/Initializer/random_uniformAdd0pi/dense_1/kernel/Initializer/random_uniform/mul0pi/dense_1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

Ż
pi/dense_1/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *$
_class
loc:@pi/dense_1/kernel
Ý
pi/dense_1/kernel/AssignAssignpi/dense_1/kernel,pi/dense_1/kernel/Initializer/random_uniform*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:


pi/dense_1/kernel/readIdentitypi/dense_1/kernel*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:


!pi/dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *"
_class
loc:@pi/dense_1/bias
Ą
pi/dense_1/bias
VariableV2*
dtype0*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
shared_name *
shape:*
	container 
Ç
pi/dense_1/bias/AssignAssignpi/dense_1/bias!pi/dense_1/bias/Initializer/zeros*
_output_shapes	
:*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(
{
pi/dense_1/bias/readIdentitypi/dense_1/bias*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:

pi/dense_1/MatMulMatMulpi/dense/Tanhpi/dense_1/kernel/read*
transpose_a( *
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

pi/dense_1/BiasAddBiasAddpi/dense_1/MatMulpi/dense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
^
pi/dense_1/TanhTanhpi/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:*$
_class
loc:@pi/dense_2/kernel

0pi/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *$
_class
loc:@pi/dense_2/kernel*
valueB
 *(ž

0pi/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *(>*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
: *
dtype0
ő
:pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_2/kernel/Initializer/random_uniform/shape*$
_class
loc:@pi/dense_2/kernel*
T0*

seed*
_output_shapes
:	*
seed2.*
dtype0
â
0pi/dense_2/kernel/Initializer/random_uniform/subSub0pi/dense_2/kernel/Initializer/random_uniform/max0pi/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
: 
ő
0pi/dense_2/kernel/Initializer/random_uniform/mulMul:pi/dense_2/kernel/Initializer/random_uniform/RandomUniform0pi/dense_2/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	
ç
,pi/dense_2/kernel/Initializer/random_uniformAdd0pi/dense_2/kernel/Initializer/random_uniform/mul0pi/dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel
­
pi/dense_2/kernel
VariableV2*
_output_shapes
:	*
shape:	*
dtype0*
shared_name *$
_class
loc:@pi/dense_2/kernel*
	container 
Ü
pi/dense_2/kernel/AssignAssignpi/dense_2/kernel,pi/dense_2/kernel/Initializer/random_uniform*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	

pi/dense_2/kernel/readIdentitypi/dense_2/kernel*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
:	

!pi/dense_2/bias/Initializer/zerosConst*
valueB*    *
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
dtype0

pi/dense_2/bias
VariableV2*
dtype0*
shared_name *"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
shape:*
	container 
Ć
pi/dense_2/bias/AssignAssignpi/dense_2/bias!pi/dense_2/bias/Initializer/zeros*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(
z
pi/dense_2/bias/readIdentitypi/dense_2/bias*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias

pi/dense_2/MatMulMatMulpi/dense_1/Tanhpi/dense_2/kernel/read*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙

pi/dense_2/BiasAddBiasAddpi/dense_2/MatMulpi/dense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
pi/log_std/initial_valueConst*
_output_shapes
:*
valueB"   ż   ż*
dtype0
v

pi/log_std
VariableV2*
_output_shapes
:*
shape:*
shared_name *
	container *
dtype0
Ž
pi/log_std/AssignAssign
pi/log_stdpi/log_std/initial_value*
use_locking(*
_output_shapes
:*
_class
loc:@pi/log_std*
validate_shape(*
T0
k
pi/log_std/readIdentity
pi/log_std*
_class
loc:@pi/log_std*
_output_shapes
:*
T0
C
pi/ExpExppi/log_std/read*
_output_shapes
:*
T0
Z
pi/ShapeShapepi/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
Z
pi/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
pi/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  ?

%pi/random_normal/RandomStandardNormalRandomStandardNormalpi/Shape*

seed*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
seed2C

pi/random_normal/mulMul%pi/random_normal/RandomStandardNormalpi/random_normal/stddev*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
pi/random_normalAddpi/random_normal/mulpi/random_normal/mean*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
pi/mulMulpi/random_normalpi/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
]
pi/addAddV2pi/dense_2/BiasAddpi/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
pi/subSubPlaceholder_1pi/dense_2/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
E
pi/Exp_1Exppi/log_std/read*
T0*
_output_shapes
:
O

pi/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
L
pi/add_1AddV2pi/Exp_1
pi/add_1/y*
_output_shapes
:*
T0
Y

pi/truedivRealDivpi/subpi/add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
M
pi/pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
U
pi/powPow
pi/truedivpi/pow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
pi/add_2AddV2pi/powpi/mul_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/add_3/yConst*
_output_shapes
: *
valueB
 *?ë?*
dtype0
Y
pi/add_3AddV2pi/add_2
pi/add_3/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/mul_2/xConst*
_output_shapes
: *
valueB
 *   ż*
dtype0
W
pi/mul_2Mul
pi/mul_2/xpi/add_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
pi/Sum/reduction_indicesConst*
value	B :*
_output_shapes
: *
dtype0
|
pi/SumSumpi/mul_2pi/Sum/reduction_indices*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *

Tidx0
]
pi/sub_1Subpi/addpi/dense_2/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
E
pi/Exp_2Exppi/log_std/read*
T0*
_output_shapes
:
O

pi/add_4/yConst*
dtype0*
valueB
 *wĚ+2*
_output_shapes
: 
L
pi/add_4AddV2pi/Exp_2
pi/add_4/y*
T0*
_output_shapes
:
]
pi/truediv_1RealDivpi/sub_1pi/add_4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/pow_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
[
pi/pow_1Powpi/truediv_1
pi/pow_1/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/mul_3/xConst*
valueB
 *   @*
_output_shapes
: *
dtype0
Q
pi/mul_3Mul
pi/mul_3/xpi/log_std/read*
_output_shapes
:*
T0
W
pi/add_5AddV2pi/pow_1pi/mul_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/add_6/yConst*
valueB
 *?ë?*
dtype0*
_output_shapes
: 
Y
pi/add_6AddV2pi/add_5
pi/add_6/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_4/xConst*
dtype0*
valueB
 *   ż*
_output_shapes
: 
W
pi/mul_4Mul
pi/mul_4/xpi/add_6*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
pi/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 

pi/Sum_1Sumpi/mul_4pi/Sum_1/reduction_indices*

Tidx0*
T0*
	keep_dims( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
pi/PlaceholderPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
pi/Placeholder_1Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙*
dtype0
O

pi/mul_5/xConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
Q
pi/mul_5Mul
pi/mul_5/xpi/log_std/read*
_output_shapes
:*
T0
>
pi/Exp_3Exppi/mul_5*
_output_shapes
:*
T0
O

pi/mul_6/xConst*
_output_shapes
: *
valueB
 *   @*
dtype0
_
pi/mul_6Mul
pi/mul_6/xpi/Placeholder_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
K
pi/Exp_4Exppi/mul_6*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
pi/sub_2Subpi/Placeholderpi/dense_2/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/pow_2/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0
W
pi/pow_2Powpi/sub_2
pi/pow_2/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
W
pi/add_7AddV2pi/pow_2pi/Exp_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/add_8/yConst*
_output_shapes
: *
valueB
 *wĚ+2*
dtype0
Y
pi/add_8AddV2pi/Exp_4
pi/add_8/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
]
pi/truediv_2RealDivpi/add_7pi/add_8*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/sub_3/yConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
[
pi/sub_3Subpi/truediv_2
pi/sub_3/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
W
pi/mul_7Mul
pi/mul_7/xpi/sub_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
_
pi/add_9AddV2pi/mul_7pi/Placeholder_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
pi/sub_4Subpi/add_9pi/log_std/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
pi/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :

pi/Sum_2Sumpi/sub_4pi/Sum_2/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *
T0*

Tidx0
R
pi/ConstConst*
valueB: *
dtype0*
_output_shapes
:
a
pi/MeanMeanpi/Sum_2pi/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
P
pi/add_10/yConst*
_output_shapes
: *
valueB
 *Çľ?*
dtype0
U
	pi/add_10AddV2pi/log_std/readpi/add_10/y*
T0*
_output_shapes
:
e
pi/Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
t
pi/Sum_3Sum	pi/add_10pi/Sum_3/reduction_indices*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
M

pi/Const_1Const*
dtype0*
valueB *
_output_shapes
: 
e
	pi/Mean_1Meanpi/Sum_3
pi/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ľ
0vf/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*"
_class
loc:@vf/dense/kernel*
dtype0*
valueB"<      

.vf/dense/kernel/Initializer/random_uniform/minConst*
dtype0*"
_class
loc:@vf/dense/kernel*
_output_shapes
: *
valueB
 *ž

.vf/dense/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@vf/dense/kernel*
valueB
 *>*
_output_shapes
: *
dtype0
đ
8vf/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vf/dense/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
seed2*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*

seed
Ú
.vf/dense/kernel/Initializer/random_uniform/subSub.vf/dense/kernel/Initializer/random_uniform/max.vf/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@vf/dense/kernel*
_output_shapes
: *
T0
í
.vf/dense/kernel/Initializer/random_uniform/mulMul8vf/dense/kernel/Initializer/random_uniform/RandomUniform.vf/dense/kernel/Initializer/random_uniform/sub*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	<
ß
*vf/dense/kernel/Initializer/random_uniformAdd.vf/dense/kernel/Initializer/random_uniform/mul.vf/dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel
Š
vf/dense/kernel
VariableV2*
shared_name *
dtype0*
	container *
shape:	<*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<
Ô
vf/dense/kernel/AssignAssignvf/dense/kernel*vf/dense/kernel/Initializer/random_uniform*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
validate_shape(*
T0*
use_locking(

vf/dense/kernel/readIdentityvf/dense/kernel*
T0*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel

vf/dense/bias/Initializer/zerosConst*
valueB*    *
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
dtype0

vf/dense/bias
VariableV2*
dtype0*
shared_name *
shape:*
_output_shapes	
:*
	container * 
_class
loc:@vf/dense/bias
ż
vf/dense/bias/AssignAssignvf/dense/biasvf/dense/bias/Initializer/zeros*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
T0
u
vf/dense/bias/readIdentityvf/dense/bias*
T0* 
_class
loc:@vf/dense/bias*
_output_shapes	
:

vf/dense/MatMulMatMulPlaceholdervf/dense/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
T0*
transpose_a( 

vf/dense/BiasAddBiasAddvf/dense/MatMulvf/dense/bias/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
Z
vf/dense/TanhTanhvf/dense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
2vf/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*$
_class
loc:@vf/dense_1/kernel*
valueB"      *
dtype0

0vf/dense_1/kernel/Initializer/random_uniform/minConst*$
_class
loc:@vf/dense_1/kernel*
valueB
 *×łÝ˝*
_output_shapes
: *
dtype0

0vf/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *×łÝ=*
_output_shapes
: *$
_class
loc:@vf/dense_1/kernel*
dtype0
÷
:vf/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_1/kernel/Initializer/random_uniform/shape*$
_class
loc:@vf/dense_1/kernel*
seed2*
dtype0*

seed*
T0* 
_output_shapes
:

â
0vf/dense_1/kernel/Initializer/random_uniform/subSub0vf/dense_1/kernel/Initializer/random_uniform/max0vf/dense_1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *$
_class
loc:@vf/dense_1/kernel
ö
0vf/dense_1/kernel/Initializer/random_uniform/mulMul:vf/dense_1/kernel/Initializer/random_uniform/RandomUniform0vf/dense_1/kernel/Initializer/random_uniform/sub*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0
č
,vf/dense_1/kernel/Initializer/random_uniformAdd0vf/dense_1/kernel/Initializer/random_uniform/mul0vf/dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*$
_class
loc:@vf/dense_1/kernel
Ż
vf/dense_1/kernel
VariableV2*
shape:
*$
_class
loc:@vf/dense_1/kernel*
shared_name *
dtype0* 
_output_shapes
:
*
	container 
Ý
vf/dense_1/kernel/AssignAssignvf/dense_1/kernel,vf/dense_1/kernel/Initializer/random_uniform*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(*
validate_shape(

vf/dense_1/kernel/readIdentityvf/dense_1/kernel* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0

!vf/dense_1/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@vf/dense_1/bias*
dtype0*
_output_shapes	
:
Ą
vf/dense_1/bias
VariableV2*
dtype0*
shape:*
shared_name *
	container *"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:
Ç
vf/dense_1/bias/AssignAssignvf/dense_1/bias!vf/dense_1/bias/Initializer/zeros*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:
{
vf/dense_1/bias/readIdentityvf/dense_1/bias*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0

vf/dense_1/MatMulMatMulvf/dense/Tanhvf/dense_1/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

vf/dense_1/BiasAddBiasAddvf/dense_1/MatMulvf/dense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
^
vf/dense_1/TanhTanhvf/dense_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
2vf/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*$
_class
loc:@vf/dense_2/kernel*
valueB"      

0vf/dense_2/kernel/Initializer/random_uniform/minConst*$
_class
loc:@vf/dense_2/kernel*
dtype0*
valueB
 *Ivž*
_output_shapes
: 

0vf/dense_2/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
: *
valueB
 *Iv>*
dtype0
ö
:vf/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_2/kernel/Initializer/random_uniform/shape*
T0*

seed*
seed2Ź*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
dtype0
â
0vf/dense_2/kernel/Initializer/random_uniform/subSub0vf/dense_2/kernel/Initializer/random_uniform/max0vf/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@vf/dense_2/kernel
ő
0vf/dense_2/kernel/Initializer/random_uniform/mulMul:vf/dense_2/kernel/Initializer/random_uniform/RandomUniform0vf/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*$
_class
loc:@vf/dense_2/kernel
ç
,vf/dense_2/kernel/Initializer/random_uniformAdd0vf/dense_2/kernel/Initializer/random_uniform/mul0vf/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0
­
vf/dense_2/kernel
VariableV2*$
_class
loc:@vf/dense_2/kernel*
	container *
shape:	*
dtype0*
shared_name *
_output_shapes
:	
Ü
vf/dense_2/kernel/AssignAssignvf/dense_2/kernel,vf/dense_2/kernel/Initializer/random_uniform*
_output_shapes
:	*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
use_locking(

vf/dense_2/kernel/readIdentityvf/dense_2/kernel*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
T0

!vf/dense_2/bias/Initializer/zerosConst*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
dtype0*
valueB*    

vf/dense_2/bias
VariableV2*
	container *
shape:*
_output_shapes
:*
dtype0*"
_class
loc:@vf/dense_2/bias*
shared_name 
Ć
vf/dense_2/bias/AssignAssignvf/dense_2/bias!vf/dense_2/bias/Initializer/zeros*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0
z
vf/dense_2/bias/readIdentityvf/dense_2/bias*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:

vf/dense_2/MatMulMatMulvf/dense_1/Tanhvf/dense_2/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

vf/dense_2/BiasAddBiasAddvf/dense_2/MatMulvf/dense_2/bias/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
n

vf/SqueezeSqueezevf/dense_2/BiasAdd*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
*
T0
Ľ
0vc/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:*
valueB"<      

.vc/dense/kernel/Initializer/random_uniform/minConst*"
_class
loc:@vc/dense/kernel*
_output_shapes
: *
valueB
 *ž*
dtype0

.vc/dense/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@vc/dense/kernel*
valueB
 *>*
_output_shapes
: *
dtype0
đ
8vc/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vc/dense/kernel/Initializer/random_uniform/shape*
T0*

seed*
seed2˝*
dtype0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<
Ú
.vc/dense/kernel/Initializer/random_uniform/subSub.vc/dense/kernel/Initializer/random_uniform/max.vc/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@vc/dense/kernel*
_output_shapes
: *
T0
í
.vc/dense/kernel/Initializer/random_uniform/mulMul8vc/dense/kernel/Initializer/random_uniform/RandomUniform.vc/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	<*
T0*"
_class
loc:@vc/dense/kernel
ß
*vc/dense/kernel/Initializer/random_uniformAdd.vc/dense/kernel/Initializer/random_uniform/mul.vc/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<
Š
vc/dense/kernel
VariableV2*
shared_name *
	container *
dtype0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
shape:	<
Ô
vc/dense/kernel/AssignAssignvc/dense/kernel*vc/dense/kernel/Initializer/random_uniform*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	<*
use_locking(

vc/dense/kernel/readIdentityvc/dense/kernel*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
T0

vc/dense/bias/Initializer/zerosConst*
dtype0* 
_class
loc:@vc/dense/bias*
valueB*    *
_output_shapes	
:

vc/dense/bias
VariableV2*
dtype0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
	container *
shape:*
shared_name 
ż
vc/dense/bias/AssignAssignvc/dense/biasvc/dense/bias/Initializer/zeros*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
validate_shape(*
use_locking(
u
vc/dense/bias/readIdentityvc/dense/bias* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0

vc/dense/MatMulMatMulPlaceholdervc/dense/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

vc/dense/BiasAddBiasAddvc/dense/MatMulvc/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
vc/dense/TanhTanhvc/dense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
2vc/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_output_shapes
:*$
_class
loc:@vc/dense_1/kernel*
dtype0

0vc/dense_1/kernel/Initializer/random_uniform/minConst*$
_class
loc:@vc/dense_1/kernel*
valueB
 *×łÝ˝*
_output_shapes
: *
dtype0

0vc/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *×łÝ=*
_output_shapes
: *
dtype0*$
_class
loc:@vc/dense_1/kernel
÷
:vc/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_1/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
seed2Î*

seed*
T0
â
0vc/dense_1/kernel/Initializer/random_uniform/subSub0vc/dense_1/kernel/Initializer/random_uniform/max0vc/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@vc/dense_1/kernel*
_output_shapes
: *
T0
ö
0vc/dense_1/kernel/Initializer/random_uniform/mulMul:vc/dense_1/kernel/Initializer/random_uniform/RandomUniform0vc/dense_1/kernel/Initializer/random_uniform/sub*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0
č
,vc/dense_1/kernel/Initializer/random_uniformAdd0vc/dense_1/kernel/Initializer/random_uniform/mul0vc/dense_1/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
Ż
vc/dense_1/kernel
VariableV2*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
shared_name *
dtype0*
	container *
shape:

Ý
vc/dense_1/kernel/AssignAssignvc/dense_1/kernel,vc/dense_1/kernel/Initializer/random_uniform* 
_output_shapes
:
*
T0*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(

vc/dense_1/kernel/readIdentityvc/dense_1/kernel*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:


!vc/dense_1/bias/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
Ą
vc/dense_1/bias
VariableV2*
shape:*
	container *
_output_shapes	
:*
shared_name *
dtype0*"
_class
loc:@vc/dense_1/bias
Ç
vc/dense_1/bias/AssignAssignvc/dense_1/bias!vc/dense_1/bias/Initializer/zeros*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
{
vc/dense_1/bias/readIdentityvc/dense_1/bias*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:

vc/dense_1/MatMulMatMulvc/dense/Tanhvc/dense_1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( *
transpose_b( 

vc/dense_1/BiasAddBiasAddvc/dense_1/MatMulvc/dense_1/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
vc/dense_1/TanhTanhvc/dense_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
2vc/dense_2/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@vc/dense_2/kernel*
dtype0*
valueB"      *
_output_shapes
:

0vc/dense_2/kernel/Initializer/random_uniform/minConst*$
_class
loc:@vc/dense_2/kernel*
dtype0*
valueB
 *Ivž*
_output_shapes
: 

0vc/dense_2/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *$
_class
loc:@vc/dense_2/kernel*
valueB
 *Iv>*
dtype0
ö
:vc/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_2/kernel/Initializer/random_uniform/shape*
T0*
seed2ß*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*

seed*
dtype0
â
0vc/dense_2/kernel/Initializer/random_uniform/subSub0vc/dense_2/kernel/Initializer/random_uniform/max0vc/dense_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
: 
ő
0vc/dense_2/kernel/Initializer/random_uniform/mulMul:vc/dense_2/kernel/Initializer/random_uniform/RandomUniform0vc/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
T0
ç
,vc/dense_2/kernel/Initializer/random_uniformAdd0vc/dense_2/kernel/Initializer/random_uniform/mul0vc/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
T0
­
vc/dense_2/kernel
VariableV2*
	container *$
_class
loc:@vc/dense_2/kernel*
shared_name *
_output_shapes
:	*
shape:	*
dtype0
Ü
vc/dense_2/kernel/AssignAssignvc/dense_2/kernel,vc/dense_2/kernel/Initializer/random_uniform*
use_locking(*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
T0

vc/dense_2/kernel/readIdentityvc/dense_2/kernel*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	

!vc/dense_2/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
dtype0

vc/dense_2/bias
VariableV2*"
_class
loc:@vc/dense_2/bias*
shape:*
	container *
_output_shapes
:*
shared_name *
dtype0
Ć
vc/dense_2/bias/AssignAssignvc/dense_2/bias!vc/dense_2/bias/Initializer/zeros*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:*
use_locking(
z
vc/dense_2/bias/readIdentityvc/dense_2/bias*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:

vc/dense_2/MatMulMatMulvc/dense_1/Tanhvc/dense_2/kernel/read*
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

vc/dense_2/BiasAddBiasAddvc/dense_2/MatMulvc/dense_2/bias/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
n

vc/SqueezeSqueezevc/dense_2/BiasAdd*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
*
T0
@
NegNegpi/Sum*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O
ConstConst*
dtype0*
valueB: *
_output_shapes
:
V
MeanMeanNegConst*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
L
Const_1Const*
dtype0*
valueB
 *D
?*
_output_shapes
: 
h
#penalty/penalty_param/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *D
?
y
penalty/penalty_param
VariableV2*
shape: *
	container *
shared_name *
_output_shapes
: *
dtype0
Ö
penalty/penalty_param/AssignAssignpenalty/penalty_param#penalty/penalty_param/initial_value*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

penalty/penalty_param/readIdentitypenalty/penalty_param*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param
Q
SoftplusSoftpluspenalty/penalty_param/read*
_output_shapes
: *
T0
I
Neg_1Negpenalty/penalty_param/read*
T0*
_output_shapes
: 
J
sub/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0
A
subSubPlaceholder_8sub/y*
T0*
_output_shapes
: 
7
mulMulNeg_1sub*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
_output_shapes
: *
dtype0
X
gradients/grad_ys_0Const*
dtype0*
valueB
 *  ?*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
S
gradients/mul_grad/MulMulgradients/Fillsub*
_output_shapes
: *
T0
W
gradients/mul_grad/Mul_1Mulgradients/FillNeg_1*
_output_shapes
: *
T0
_
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Mul^gradients/mul_grad/Mul_1
Á
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Mul$^gradients/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*)
_class
loc:@gradients/mul_grad/Mul
Ç
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Mul_1$^gradients/mul_grad/tuple/group_deps*+
_class!
loc:@gradients/mul_grad/Mul_1*
_output_shapes
: *
T0
m
gradients/Neg_1_grad/NegNeg+gradients/mul_grad/tuple/control_dependency*
_output_shapes
: *
T0
`
Reshape/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
n
ReshapeReshapegradients/Neg_1_grad/NegReshape/shape*
Tshape0*
T0*
_output_shapes
:
S
concat/concat_dimConst*
value	B : *
_output_shapes
: *
dtype0
G
concat/concatIdentityReshape*
_output_shapes
:*
T0
m
PyFuncPyFuncconcat/concat*
token
pyfunc_0*
_output_shapes
:*
Tin
2*
Tout
2
Q
Const_2Const*
dtype0*
valueB:*
_output_shapes
:
Q
split/split_dimConst*
_output_shapes
: *
value	B : *
dtype0
s
splitSplitVPyFuncConst_2split/split_dim*
T0*

Tlen0*
	num_split*
_output_shapes
:
R
Reshape_1/shapeConst*
_output_shapes
: *
valueB *
dtype0
[
	Reshape_1ReshapesplitReshape_1/shape*
Tshape0*
T0*
_output_shapes
: 

beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param

beta1_power
VariableV2*
shared_name *
shape: *
	container *(
_class
loc:@penalty/penalty_param*
dtype0*
_output_shapes
: 
¸
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(
t
beta1_power/readIdentitybeta1_power*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param

beta2_power/initial_valueConst*
_output_shapes
: *
valueB
 *wž?*(
_class
loc:@penalty/penalty_param*
dtype0

beta2_power
VariableV2*
shape: *
	container *(
_class
loc:@penalty/penalty_param*
shared_name *
dtype0*
_output_shapes
: 
¸
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: *
T0
t
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param

,penalty/penalty_param/Adam/Initializer/zerosConst*
_output_shapes
: *
valueB
 *    *
dtype0*(
_class
loc:@penalty/penalty_param
¨
penalty/penalty_param/Adam
VariableV2*
shape: *
_output_shapes
: *
shared_name *
	container *(
_class
loc:@penalty/penalty_param*
dtype0
é
!penalty/penalty_param/Adam/AssignAssignpenalty/penalty_param/Adam,penalty/penalty_param/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: *
T0

penalty/penalty_param/Adam/readIdentitypenalty/penalty_param/Adam*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0

.penalty/penalty_param/Adam_1/Initializer/zerosConst*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
valueB
 *    *
dtype0
Ş
penalty/penalty_param/Adam_1
VariableV2*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
ď
#penalty/penalty_param/Adam_1/AssignAssignpenalty/penalty_param/Adam_1.penalty/penalty_param/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: 

!penalty/penalty_param/Adam_1/readIdentitypenalty/penalty_param/Adam_1*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *ÍĚL=*
dtype0
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
dtype0*
valueB
 *wž?*
_output_shapes
: 
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *wĚ+2*
dtype0
ä
+Adam/update_penalty/penalty_param/ApplyAdam	ApplyAdampenalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_1*
_output_shapes
: *
use_locking( *(
_class
loc:@penalty/penalty_param*
use_nesterov( *
T0
Ś
Adam/mulMulbeta1_power/read
Adam/beta1,^Adam/update_penalty/penalty_param/ApplyAdam*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0
 
Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *
use_locking( *
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0
¨

Adam/mul_1Mulbeta2_power/read
Adam/beta2,^Adam/update_penalty/penalty_param/ApplyAdam*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
¤
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(*
use_locking( 
X
AdamNoOp^Adam/Assign^Adam/Assign_1,^Adam/update_penalty/penalty_param/ApplyAdam
i
Reshape_2/shapeConst^Adam*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
t
	Reshape_2Reshapepenalty/penalty_param/readReshape_2/shape*
T0*
Tshape0*
_output_shapes
:
\
concat_1/concat_dimConst^Adam*
_output_shapes
: *
dtype0*
value	B : 
K
concat_1/concatIdentity	Reshape_2*
T0*
_output_shapes
:
o
PyFunc_1PyFuncconcat_1/concat*
Tin
2*
token
pyfunc_1*
_output_shapes
:*
Tout
2
X
Const_3Const^Adam*
valueB:*
_output_shapes
:*
dtype0
Z
split_1/split_dimConst^Adam*
dtype0*
_output_shapes
: *
value	B : 
w
split_1SplitVPyFunc_1Const_3split_1/split_dim*
T0*

Tlen0*
	num_split*
_output_shapes
:
Y
Reshape_3/shapeConst^Adam*
_output_shapes
: *
valueB *
dtype0
]
	Reshape_3Reshapesplit_1Reshape_3/shape*
T0*
Tshape0*
_output_shapes
: 
Ś
AssignAssignpenalty/penalty_param	Reshape_3*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
"

group_depsNoOp^Adam^Assign
(
group_deps_1NoOp^Adam^group_deps
Q
sub_1Subpi/SumPlaceholder_6*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
?
ExpExpsub_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
N
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
Z
GreaterGreaterPlaceholder_2	Greater/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
L
mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *?
R
mul_1Mulmul_1/xPlaceholder_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL?
R
mul_2Mulmul_2/xPlaceholder_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
U
SelectSelectGreatermul_1mul_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
N
mul_3MulExpPlaceholder_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O
MinimumMinimummul_3Select*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_4Const*
valueB: *
_output_shapes
:*
dtype0
^
Mean_1MeanMinimumConst_4*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
N
mul_4MulExpPlaceholder_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_5Const*
_output_shapes
:*
valueB: *
dtype0
\
Mean_2Meanmul_4Const_5*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
L
mul_5/xConst*
_output_shapes
: *
valueB
 *    *
dtype0
A
mul_5Mulmul_5/x	pi/Mean_1*
T0*
_output_shapes
: 
<
addAddV2Mean_1mul_5*
T0*
_output_shapes
: 
?
mul_6MulSoftplusMean_2*
_output_shapes
: *
T0
9
sub_2Subaddmul_6*
T0*
_output_shapes
: 
L
add_1/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
B
add_1AddV2add_1/xSoftplus*
T0*
_output_shapes
: 
A
truedivRealDivsub_2add_1*
T0*
_output_shapes
: 
6
Neg_2Negtruediv*
_output_shapes
: *
T0
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
_output_shapes
: *
T0*

index_type0
T
gradients_1/Neg_2_grad/NegNeggradients_1/Fill*
_output_shapes
: *
T0
a
gradients_1/truediv_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
c
 gradients_1/truediv_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
Ć
.gradients_1/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/truediv_grad/Shape gradients_1/truediv_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
o
 gradients_1/truediv_grad/RealDivRealDivgradients_1/Neg_2_grad/Negadd_1*
T0*
_output_shapes
: 
ł
gradients_1/truediv_grad/SumSum gradients_1/truediv_grad/RealDiv.gradients_1/truediv_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

 gradients_1/truediv_grad/ReshapeReshapegradients_1/truediv_grad/Sumgradients_1/truediv_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
K
gradients_1/truediv_grad/NegNegsub_2*
_output_shapes
: *
T0
s
"gradients_1/truediv_grad/RealDiv_1RealDivgradients_1/truediv_grad/Negadd_1*
_output_shapes
: *
T0
y
"gradients_1/truediv_grad/RealDiv_2RealDiv"gradients_1/truediv_grad/RealDiv_1add_1*
_output_shapes
: *
T0

gradients_1/truediv_grad/mulMulgradients_1/Neg_2_grad/Neg"gradients_1/truediv_grad/RealDiv_2*
T0*
_output_shapes
: 
ł
gradients_1/truediv_grad/Sum_1Sumgradients_1/truediv_grad/mul0gradients_1/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 

"gradients_1/truediv_grad/Reshape_1Reshapegradients_1/truediv_grad/Sum_1 gradients_1/truediv_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
y
)gradients_1/truediv_grad/tuple/group_depsNoOp!^gradients_1/truediv_grad/Reshape#^gradients_1/truediv_grad/Reshape_1
á
1gradients_1/truediv_grad/tuple/control_dependencyIdentity gradients_1/truediv_grad/Reshape*^gradients_1/truediv_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/truediv_grad/Reshape*
_output_shapes
: *
T0
ç
3gradients_1/truediv_grad/tuple/control_dependency_1Identity"gradients_1/truediv_grad/Reshape_1*^gradients_1/truediv_grad/tuple/group_deps*
T0*
_output_shapes
: *5
_class+
)'loc:@gradients_1/truediv_grad/Reshape_1
u
gradients_1/sub_2_grad/NegNeg1gradients_1/truediv_grad/tuple/control_dependency*
T0*
_output_shapes
: 

'gradients_1/sub_2_grad/tuple/group_depsNoOp^gradients_1/sub_2_grad/Neg2^gradients_1/truediv_grad/tuple/control_dependency
î
/gradients_1/sub_2_grad/tuple/control_dependencyIdentity1gradients_1/truediv_grad/tuple/control_dependency(^gradients_1/sub_2_grad/tuple/group_deps*
_output_shapes
: *3
_class)
'%loc:@gradients_1/truediv_grad/Reshape*
T0
Ó
1gradients_1/sub_2_grad/tuple/control_dependency_1Identitygradients_1/sub_2_grad/Neg(^gradients_1/sub_2_grad/tuple/group_deps*-
_class#
!loc:@gradients_1/sub_2_grad/Neg*
T0*
_output_shapes
: 
e
'gradients_1/add_1_grad/tuple/group_depsNoOp4^gradients_1/truediv_grad/tuple/control_dependency_1
ň
/gradients_1/add_1_grad/tuple/control_dependencyIdentity3gradients_1/truediv_grad/tuple/control_dependency_1(^gradients_1/add_1_grad/tuple/group_deps*5
_class+
)'loc:@gradients_1/truediv_grad/Reshape_1*
T0*
_output_shapes
: 
ô
1gradients_1/add_1_grad/tuple/control_dependency_1Identity3gradients_1/truediv_grad/tuple/control_dependency_1(^gradients_1/add_1_grad/tuple/group_deps*5
_class+
)'loc:@gradients_1/truediv_grad/Reshape_1*
_output_shapes
: *
T0
_
%gradients_1/add_grad/tuple/group_depsNoOp0^gradients_1/sub_2_grad/tuple/control_dependency
č
-gradients_1/add_grad/tuple/control_dependencyIdentity/gradients_1/sub_2_grad/tuple/control_dependency&^gradients_1/add_grad/tuple/group_deps*
_output_shapes
: *3
_class)
'%loc:@gradients_1/truediv_grad/Reshape*
T0
ę
/gradients_1/add_grad/tuple/control_dependency_1Identity/gradients_1/sub_2_grad/tuple/control_dependency&^gradients_1/add_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/truediv_grad/Reshape*
T0*
_output_shapes
: 
}
gradients_1/mul_6_grad/MulMul1gradients_1/sub_2_grad/tuple/control_dependency_1Mean_2*
T0*
_output_shapes
: 

gradients_1/mul_6_grad/Mul_1Mul1gradients_1/sub_2_grad/tuple/control_dependency_1Softplus*
T0*
_output_shapes
: 
k
'gradients_1/mul_6_grad/tuple/group_depsNoOp^gradients_1/mul_6_grad/Mul^gradients_1/mul_6_grad/Mul_1
Ń
/gradients_1/mul_6_grad/tuple/control_dependencyIdentitygradients_1/mul_6_grad/Mul(^gradients_1/mul_6_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients_1/mul_6_grad/Mul*
_output_shapes
: 
×
1gradients_1/mul_6_grad/tuple/control_dependency_1Identitygradients_1/mul_6_grad/Mul_1(^gradients_1/mul_6_grad/tuple/group_deps*
T0*
_output_shapes
: */
_class%
#!loc:@gradients_1/mul_6_grad/Mul_1
o
%gradients_1/Mean_1_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
ł
gradients_1/Mean_1_grad/ReshapeReshape-gradients_1/add_grad/tuple/control_dependency%gradients_1/Mean_1_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
d
gradients_1/Mean_1_grad/ShapeShapeMinimum*
_output_shapes
:*
out_type0*
T0
¤
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*

Tmultiples0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
f
gradients_1/Mean_1_grad/Shape_1ShapeMinimum*
T0*
_output_shapes
:*
out_type0
b
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
g
gradients_1/Mean_1_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
˘
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
i
gradients_1/Mean_1_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
Ś
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
c
!gradients_1/Mean_1_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :

gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0

 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
T0*
_output_shapes
: 

gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0*
Truncate( 

gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
~
gradients_1/mul_5_grad/MulMul/gradients_1/add_grad/tuple/control_dependency_1	pi/Mean_1*
T0*
_output_shapes
: 
~
gradients_1/mul_5_grad/Mul_1Mul/gradients_1/add_grad/tuple/control_dependency_1mul_5/x*
T0*
_output_shapes
: 
k
'gradients_1/mul_5_grad/tuple/group_depsNoOp^gradients_1/mul_5_grad/Mul^gradients_1/mul_5_grad/Mul_1
Ń
/gradients_1/mul_5_grad/tuple/control_dependencyIdentitygradients_1/mul_5_grad/Mul(^gradients_1/mul_5_grad/tuple/group_deps*
T0*
_output_shapes
: *-
_class#
!loc:@gradients_1/mul_5_grad/Mul
×
1gradients_1/mul_5_grad/tuple/control_dependency_1Identitygradients_1/mul_5_grad/Mul_1(^gradients_1/mul_5_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/mul_5_grad/Mul_1*
_output_shapes
: 
Ý
gradients_1/AddNAddN1gradients_1/add_1_grad/tuple/control_dependency_1/gradients_1/mul_6_grad/tuple/control_dependency*
_output_shapes
: *5
_class+
)'loc:@gradients_1/truediv_grad/Reshape_1*
N*
T0
i
!gradients_1/Softplus_grad/SigmoidSigmoidpenalty/penalty_param/read*
_output_shapes
: *
T0
z
gradients_1/Softplus_grad/mulMulgradients_1/AddN!gradients_1/Softplus_grad/Sigmoid*
T0*
_output_shapes
: 
o
%gradients_1/Mean_2_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
ˇ
gradients_1/Mean_2_grad/ReshapeReshape1gradients_1/mul_6_grad/tuple/control_dependency_1%gradients_1/Mean_2_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
b
gradients_1/Mean_2_grad/ShapeShapemul_4*
out_type0*
_output_shapes
:*
T0
¤
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
gradients_1/Mean_2_grad/Shape_1Shapemul_4*
_output_shapes
:*
T0*
out_type0
b
gradients_1/Mean_2_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
g
gradients_1/Mean_2_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
˘
gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
i
gradients_1/Mean_2_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
Ś
gradients_1/Mean_2_grad/Prod_1Prodgradients_1/Mean_2_grad/Shape_2gradients_1/Mean_2_grad/Const_1*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
c
!gradients_1/Mean_2_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :

gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
_output_shapes
: *
T0

 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
_output_shapes
: *
T0

gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*
Truncate( *

DstT0*

SrcT0*
_output_shapes
: 

gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
c
gradients_1/Minimum_grad/ShapeShapemul_3*
out_type0*
_output_shapes
:*
T0
f
 gradients_1/Minimum_grad/Shape_1ShapeSelect*
out_type0*
_output_shapes
:*
T0

 gradients_1/Minimum_grad/Shape_2Shapegradients_1/Mean_1_grad/truediv*
out_type0*
_output_shapes
:*
T0
i
$gradients_1/Minimum_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ž
gradients_1/Minimum_grad/zerosFill gradients_1/Minimum_grad/Shape_2$gradients_1/Minimum_grad/zeros/Const*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

index_type0
l
"gradients_1/Minimum_grad/LessEqual	LessEqualmul_3Select*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
.gradients_1/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Minimum_grad/Shape gradients_1/Minimum_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ź
gradients_1/Minimum_grad/SelectSelect"gradients_1/Minimum_grad/LessEqualgradients_1/Mean_1_grad/truedivgradients_1/Minimum_grad/zeros*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients_1/Minimum_grad/SumSumgradients_1/Minimum_grad/Select.gradients_1/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ľ
 gradients_1/Minimum_grad/ReshapeReshapegradients_1/Minimum_grad/Sumgradients_1/Minimum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ž
!gradients_1/Minimum_grad/Select_1Select"gradients_1/Minimum_grad/LessEqualgradients_1/Minimum_grad/zerosgradients_1/Mean_1_grad/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
gradients_1/Minimum_grad/Sum_1Sum!gradients_1/Minimum_grad/Select_10gradients_1/Minimum_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
Ť
"gradients_1/Minimum_grad/Reshape_1Reshapegradients_1/Minimum_grad/Sum_1 gradients_1/Minimum_grad/Shape_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
y
)gradients_1/Minimum_grad/tuple/group_depsNoOp!^gradients_1/Minimum_grad/Reshape#^gradients_1/Minimum_grad/Reshape_1
î
1gradients_1/Minimum_grad/tuple/control_dependencyIdentity gradients_1/Minimum_grad/Reshape*^gradients_1/Minimum_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*3
_class)
'%loc:@gradients_1/Minimum_grad/Reshape*
T0
ô
3gradients_1/Minimum_grad/tuple/control_dependency_1Identity"gradients_1/Minimum_grad/Reshape_1*^gradients_1/Minimum_grad/tuple/group_deps*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*5
_class+
)'loc:@gradients_1/Minimum_grad/Reshape_1
k
(gradients_1/pi/Mean_1_grad/Reshape/shapeConst*
dtype0*
valueB *
_output_shapes
: 
š
"gradients_1/pi/Mean_1_grad/ReshapeReshape1gradients_1/mul_5_grad/tuple/control_dependency_1(gradients_1/pi/Mean_1_grad/Reshape/shape*
_output_shapes
: *
Tshape0*
T0
c
 gradients_1/pi/Mean_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
 
gradients_1/pi/Mean_1_grad/TileTile"gradients_1/pi/Mean_1_grad/Reshape gradients_1/pi/Mean_1_grad/Const*
_output_shapes
: *

Tmultiples0*
T0
g
"gradients_1/pi/Mean_1_grad/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

"gradients_1/pi/Mean_1_grad/truedivRealDivgradients_1/pi/Mean_1_grad/Tile"gradients_1/pi/Mean_1_grad/Const_1*
T0*
_output_shapes
: 
_
gradients_1/mul_4_grad/ShapeShapeExp*
_output_shapes
:*
out_type0*
T0
k
gradients_1/mul_4_grad/Shape_1ShapePlaceholder_3*
T0*
_output_shapes
:*
out_type0
Ŕ
,gradients_1/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_4_grad/Shapegradients_1/mul_4_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients_1/mul_4_grad/MulMulgradients_1/Mean_2_grad/truedivPlaceholder_3*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
gradients_1/mul_4_grad/SumSumgradients_1/mul_4_grad/Mul,gradients_1/mul_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients_1/mul_4_grad/ReshapeReshapegradients_1/mul_4_grad/Sumgradients_1/mul_4_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
w
gradients_1/mul_4_grad/Mul_1MulExpgradients_1/Mean_2_grad/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
gradients_1/mul_4_grad/Sum_1Sumgradients_1/mul_4_grad/Mul_1.gradients_1/mul_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ľ
 gradients_1/mul_4_grad/Reshape_1Reshapegradients_1/mul_4_grad/Sum_1gradients_1/mul_4_grad/Shape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
s
'gradients_1/mul_4_grad/tuple/group_depsNoOp^gradients_1/mul_4_grad/Reshape!^gradients_1/mul_4_grad/Reshape_1
ć
/gradients_1/mul_4_grad/tuple/control_dependencyIdentitygradients_1/mul_4_grad/Reshape(^gradients_1/mul_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/mul_4_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
1gradients_1/mul_4_grad/tuple/control_dependency_1Identity gradients_1/mul_4_grad/Reshape_1(^gradients_1/mul_4_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*3
_class)
'%loc:@gradients_1/mul_4_grad/Reshape_1
_
gradients_1/mul_3_grad/ShapeShapeExp*
out_type0*
T0*
_output_shapes
:
k
gradients_1/mul_3_grad/Shape_1ShapePlaceholder_2*
out_type0*
_output_shapes
:*
T0
Ŕ
,gradients_1/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_3_grad/Shapegradients_1/mul_3_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/mul_3_grad/MulMul1gradients_1/Minimum_grad/tuple/control_dependencyPlaceholder_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
gradients_1/mul_3_grad/SumSumgradients_1/mul_3_grad/Mul,gradients_1/mul_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:

gradients_1/mul_3_grad/ReshapeReshapegradients_1/mul_3_grad/Sumgradients_1/mul_3_grad/Shape*
Tshape0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/mul_3_grad/Mul_1MulExp1gradients_1/Minimum_grad/tuple/control_dependency*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ą
gradients_1/mul_3_grad/Sum_1Sumgradients_1/mul_3_grad/Mul_1.gradients_1/mul_3_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Ľ
 gradients_1/mul_3_grad/Reshape_1Reshapegradients_1/mul_3_grad/Sum_1gradients_1/mul_3_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_1/mul_3_grad/tuple/group_depsNoOp^gradients_1/mul_3_grad/Reshape!^gradients_1/mul_3_grad/Reshape_1
ć
/gradients_1/mul_3_grad/tuple/control_dependencyIdentitygradients_1/mul_3_grad/Reshape(^gradients_1/mul_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/mul_3_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ě
1gradients_1/mul_3_grad/tuple/control_dependency_1Identity gradients_1/mul_3_grad/Reshape_1(^gradients_1/mul_3_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/mul_3_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
j
 gradients_1/pi/Sum_3_grad/Cast/xConst*
dtype0*
_output_shapes
:*
valueB:
u
"gradients_1/pi/Sum_3_grad/Cast_1/xConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
`
gradients_1/pi/Sum_3_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :

gradients_1/pi/Sum_3_grad/addAddV2"gradients_1/pi/Sum_3_grad/Cast_1/xgradients_1/pi/Sum_3_grad/Size*
_output_shapes
:*
T0

gradients_1/pi/Sum_3_grad/modFloorModgradients_1/pi/Sum_3_grad/addgradients_1/pi/Sum_3_grad/Size*
_output_shapes
:*
T0
i
gradients_1/pi/Sum_3_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
g
%gradients_1/pi/Sum_3_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0
g
%gradients_1/pi/Sum_3_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
ž
gradients_1/pi/Sum_3_grad/rangeRange%gradients_1/pi/Sum_3_grad/range/startgradients_1/pi/Sum_3_grad/Size%gradients_1/pi/Sum_3_grad/range/delta*

Tidx0*
_output_shapes
:
f
$gradients_1/pi/Sum_3_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
¤
gradients_1/pi/Sum_3_grad/FillFillgradients_1/pi/Sum_3_grad/Shape$gradients_1/pi/Sum_3_grad/Fill/value*
_output_shapes
:*
T0*

index_type0
č
'gradients_1/pi/Sum_3_grad/DynamicStitchDynamicStitchgradients_1/pi/Sum_3_grad/rangegradients_1/pi/Sum_3_grad/mod gradients_1/pi/Sum_3_grad/Cast/xgradients_1/pi/Sum_3_grad/Fill*
_output_shapes
:*
N*
T0
m
#gradients_1/pi/Sum_3_grad/Maximum/xConst*
_output_shapes
:*
valueB:*
dtype0
e
#gradients_1/pi/Sum_3_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :

!gradients_1/pi/Sum_3_grad/MaximumMaximum#gradients_1/pi/Sum_3_grad/Maximum/x#gradients_1/pi/Sum_3_grad/Maximum/y*
_output_shapes
:*
T0
n
$gradients_1/pi/Sum_3_grad/floordiv/xConst*
dtype0*
valueB:*
_output_shapes
:

"gradients_1/pi/Sum_3_grad/floordivFloorDiv$gradients_1/pi/Sum_3_grad/floordiv/x!gradients_1/pi/Sum_3_grad/Maximum*
T0*
_output_shapes
:
q
'gradients_1/pi/Sum_3_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
Ź
!gradients_1/pi/Sum_3_grad/ReshapeReshape"gradients_1/pi/Mean_1_grad/truediv'gradients_1/pi/Sum_3_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
r
(gradients_1/pi/Sum_3_grad/Tile/multiplesConst*
valueB:*
dtype0*
_output_shapes
:
Ş
gradients_1/pi/Sum_3_grad/TileTile!gradients_1/pi/Sum_3_grad/Reshape(gradients_1/pi/Sum_3_grad/Tile/multiples*
T0*
_output_shapes
:*

Tmultiples0
ć
gradients_1/AddN_1AddN/gradients_1/mul_4_grad/tuple/control_dependency/gradients_1/mul_3_grad/tuple/control_dependency*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*1
_class'
%#loc:@gradients_1/mul_4_grad/Reshape*
T0*
N
f
gradients_1/Exp_grad/mulMulgradients_1/AddN_1Exp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
3gradients_1/pi/add_10_grad/BroadcastGradientArgs/s0Const*
dtype0*
_output_shapes
:*
valueB:
v
3gradients_1/pi/add_10_grad/BroadcastGradientArgs/s1Const*
valueB *
_output_shapes
: *
dtype0
đ
0gradients_1/pi/add_10_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients_1/pi/add_10_grad/BroadcastGradientArgs/s03gradients_1/pi/add_10_grad/BroadcastGradientArgs/s1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
z
0gradients_1/pi/add_10_grad/Sum/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0
ľ
gradients_1/pi/add_10_grad/SumSumgradients_1/pi/Sum_3_grad/Tile0gradients_1/pi/add_10_grad/Sum/reduction_indices*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
k
(gradients_1/pi/add_10_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0
Ś
"gradients_1/pi/add_10_grad/ReshapeReshapegradients_1/pi/add_10_grad/Sum(gradients_1/pi/add_10_grad/Reshape/shape*
_output_shapes
: *
Tshape0*
T0
y
+gradients_1/pi/add_10_grad/tuple/group_depsNoOp^gradients_1/pi/Sum_3_grad/Tile#^gradients_1/pi/add_10_grad/Reshape
ĺ
3gradients_1/pi/add_10_grad/tuple/control_dependencyIdentitygradients_1/pi/Sum_3_grad/Tile,^gradients_1/pi/add_10_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/pi/Sum_3_grad/Tile*
_output_shapes
:*
T0
ë
5gradients_1/pi/add_10_grad/tuple/control_dependency_1Identity"gradients_1/pi/add_10_grad/Reshape,^gradients_1/pi/add_10_grad/tuple/group_deps*5
_class+
)'loc:@gradients_1/pi/add_10_grad/Reshape*
T0*
_output_shapes
: 
b
gradients_1/sub_1_grad/ShapeShapepi/Sum*
T0*
out_type0*
_output_shapes
:
k
gradients_1/sub_1_grad/Shape_1ShapePlaceholder_6*
out_type0*
_output_shapes
:*
T0
Ŕ
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Š
gradients_1/sub_1_grad/SumSumgradients_1/Exp_grad/mul,gradients_1/sub_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
i
gradients_1/sub_1_grad/NegNeggradients_1/Exp_grad/mul*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
gradients_1/sub_1_grad/Sum_1Sumgradients_1/sub_1_grad/Neg.gradients_1/sub_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ľ
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Sum_1gradients_1/sub_1_grad/Shape_1*
Tshape0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
ć
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ě
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1
e
gradients_1/pi/Sum_grad/ShapeShapepi/mul_2*
_output_shapes
:*
out_type0*
T0

gradients_1/pi/Sum_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape
Ż
gradients_1/pi/Sum_grad/addAddV2pi/Sum/reduction_indicesgradients_1/pi/Sum_grad/Size*
_output_shapes
: *
T0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape
ľ
gradients_1/pi/Sum_grad/modFloorModgradients_1/pi/Sum_grad/addgradients_1/pi/Sum_grad/Size*
T0*
_output_shapes
: *0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape

gradients_1/pi/Sum_grad/Shape_1Const*
_output_shapes
: *
valueB *0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
dtype0

#gradients_1/pi/Sum_grad/range/startConst*
value	B : *
_output_shapes
: *0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
dtype0

#gradients_1/pi/Sum_grad/range/deltaConst*
_output_shapes
: *
dtype0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
value	B :
č
gradients_1/pi/Sum_grad/rangeRange#gradients_1/pi/Sum_grad/range/startgradients_1/pi/Sum_grad/Size#gradients_1/pi/Sum_grad/range/delta*

Tidx0*
_output_shapes
:*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape

"gradients_1/pi/Sum_grad/Fill/valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
value	B :
Î
gradients_1/pi/Sum_grad/FillFillgradients_1/pi/Sum_grad/Shape_1"gradients_1/pi/Sum_grad/Fill/value*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
_output_shapes
: *

index_type0*
T0

%gradients_1/pi/Sum_grad/DynamicStitchDynamicStitchgradients_1/pi/Sum_grad/rangegradients_1/pi/Sum_grad/modgradients_1/pi/Sum_grad/Shapegradients_1/pi/Sum_grad/Fill*
_output_shapes
:*
N*
T0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape

!gradients_1/pi/Sum_grad/Maximum/yConst*
_output_shapes
: *
dtype0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
value	B :
Ë
gradients_1/pi/Sum_grad/MaximumMaximum%gradients_1/pi/Sum_grad/DynamicStitch!gradients_1/pi/Sum_grad/Maximum/y*
T0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
_output_shapes
:
Ă
 gradients_1/pi/Sum_grad/floordivFloorDivgradients_1/pi/Sum_grad/Shapegradients_1/pi/Sum_grad/Maximum*
T0*0
_class&
$"loc:@gradients_1/pi/Sum_grad/Shape*
_output_shapes
:
Ë
gradients_1/pi/Sum_grad/ReshapeReshape/gradients_1/sub_1_grad/tuple/control_dependency%gradients_1/pi/Sum_grad/DynamicStitch*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
Ť
gradients_1/pi/Sum_grad/TileTilegradients_1/pi/Sum_grad/Reshape gradients_1/pi/Sum_grad/floordiv*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0
g
gradients_1/pi/mul_2_grad/ShapeShape
pi/mul_2/x*
_output_shapes
: *
T0*
out_type0
i
!gradients_1/pi/mul_2_grad/Shape_1Shapepi/add_3*
out_type0*
_output_shapes
:*
T0
É
/gradients_1/pi/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/mul_2_grad/Shape!gradients_1/pi/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
~
gradients_1/pi/mul_2_grad/MulMulgradients_1/pi/Sum_grad/Tilepi/add_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
´
gradients_1/pi/mul_2_grad/SumSumgradients_1/pi/mul_2_grad/Mul/gradients_1/pi/mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

!gradients_1/pi/mul_2_grad/ReshapeReshapegradients_1/pi/mul_2_grad/Sumgradients_1/pi/mul_2_grad/Shape*
Tshape0*
_output_shapes
: *
T0

gradients_1/pi/mul_2_grad/Mul_1Mul
pi/mul_2/xgradients_1/pi/Sum_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
gradients_1/pi/mul_2_grad/Sum_1Sumgradients_1/pi/mul_2_grad/Mul_11gradients_1/pi/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
˛
#gradients_1/pi/mul_2_grad/Reshape_1Reshapegradients_1/pi/mul_2_grad/Sum_1!gradients_1/pi/mul_2_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
|
*gradients_1/pi/mul_2_grad/tuple/group_depsNoOp"^gradients_1/pi/mul_2_grad/Reshape$^gradients_1/pi/mul_2_grad/Reshape_1
ĺ
2gradients_1/pi/mul_2_grad/tuple/control_dependencyIdentity!gradients_1/pi/mul_2_grad/Reshape+^gradients_1/pi/mul_2_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/pi/mul_2_grad/Reshape*
_output_shapes
: *
T0
ü
4gradients_1/pi/mul_2_grad/tuple/control_dependency_1Identity#gradients_1/pi/mul_2_grad/Reshape_1+^gradients_1/pi/mul_2_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/pi/mul_2_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
gradients_1/pi/add_3_grad/ShapeShapepi/add_2*
out_type0*
T0*
_output_shapes
:
i
!gradients_1/pi/add_3_grad/Shape_1Shape
pi/add_3/y*
_output_shapes
: *
out_type0*
T0
É
/gradients_1/pi/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/add_3_grad/Shape!gradients_1/pi/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ë
gradients_1/pi/add_3_grad/SumSum4gradients_1/pi/mul_2_grad/tuple/control_dependency_1/gradients_1/pi/add_3_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ź
!gradients_1/pi/add_3_grad/ReshapeReshapegradients_1/pi/add_3_grad/Sumgradients_1/pi/add_3_grad/Shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
gradients_1/pi/add_3_grad/Sum_1Sum4gradients_1/pi/mul_2_grad/tuple/control_dependency_11gradients_1/pi/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
Ą
#gradients_1/pi/add_3_grad/Reshape_1Reshapegradients_1/pi/add_3_grad/Sum_1!gradients_1/pi/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
|
*gradients_1/pi/add_3_grad/tuple/group_depsNoOp"^gradients_1/pi/add_3_grad/Reshape$^gradients_1/pi/add_3_grad/Reshape_1
ö
2gradients_1/pi/add_3_grad/tuple/control_dependencyIdentity!gradients_1/pi/add_3_grad/Reshape+^gradients_1/pi/add_3_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/pi/add_3_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
4gradients_1/pi/add_3_grad/tuple/control_dependency_1Identity#gradients_1/pi/add_3_grad/Reshape_1+^gradients_1/pi/add_3_grad/tuple/group_deps*6
_class,
*(loc:@gradients_1/pi/add_3_grad/Reshape_1*
T0*
_output_shapes
: 
e
gradients_1/pi/add_2_grad/ShapeShapepi/pow*
out_type0*
T0*
_output_shapes
:
i
!gradients_1/pi/add_2_grad/Shape_1Shapepi/mul_1*
_output_shapes
:*
out_type0*
T0
É
/gradients_1/pi/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/add_2_grad/Shape!gradients_1/pi/add_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
É
gradients_1/pi/add_2_grad/SumSum2gradients_1/pi/add_3_grad/tuple/control_dependency/gradients_1/pi/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
Ź
!gradients_1/pi/add_2_grad/ReshapeReshapegradients_1/pi/add_2_grad/Sumgradients_1/pi/add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
gradients_1/pi/add_2_grad/Sum_1Sum2gradients_1/pi/add_3_grad/tuple/control_dependency1gradients_1/pi/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ľ
#gradients_1/pi/add_2_grad/Reshape_1Reshapegradients_1/pi/add_2_grad/Sum_1!gradients_1/pi/add_2_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
|
*gradients_1/pi/add_2_grad/tuple/group_depsNoOp"^gradients_1/pi/add_2_grad/Reshape$^gradients_1/pi/add_2_grad/Reshape_1
ö
2gradients_1/pi/add_2_grad/tuple/control_dependencyIdentity!gradients_1/pi/add_2_grad/Reshape+^gradients_1/pi/add_2_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/pi/add_2_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ď
4gradients_1/pi/add_2_grad/tuple/control_dependency_1Identity#gradients_1/pi/add_2_grad/Reshape_1+^gradients_1/pi/add_2_grad/tuple/group_deps*
T0*
_output_shapes
:*6
_class,
*(loc:@gradients_1/pi/add_2_grad/Reshape_1
g
gradients_1/pi/pow_grad/ShapeShape
pi/truediv*
T0*
_output_shapes
:*
out_type0
e
gradients_1/pi/pow_grad/Shape_1Shapepi/pow/y*
T0*
out_type0*
_output_shapes
: 
Ă
-gradients_1/pi/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/pow_grad/Shapegradients_1/pi/pow_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pi/pow_grad/mulMul2gradients_1/pi/add_2_grad/tuple/control_dependencypi/pow/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
gradients_1/pi/pow_grad/sub/yConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
l
gradients_1/pi/pow_grad/subSubpi/pow/ygradients_1/pi/pow_grad/sub/y*
_output_shapes
: *
T0
}
gradients_1/pi/pow_grad/PowPow
pi/truedivgradients_1/pi/pow_grad/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/pi/pow_grad/mul_1Mulgradients_1/pi/pow_grad/mulgradients_1/pi/pow_grad/Pow*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
°
gradients_1/pi/pow_grad/SumSumgradients_1/pi/pow_grad/mul_1-gradients_1/pi/pow_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ś
gradients_1/pi/pow_grad/ReshapeReshapegradients_1/pi/pow_grad/Sumgradients_1/pi/pow_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
f
!gradients_1/pi/pow_grad/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients_1/pi/pow_grad/GreaterGreater
pi/truediv!gradients_1/pi/pow_grad/Greater/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
'gradients_1/pi/pow_grad/ones_like/ShapeShape
pi/truediv*
_output_shapes
:*
T0*
out_type0
l
'gradients_1/pi/pow_grad/ones_like/ConstConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
ż
!gradients_1/pi/pow_grad/ones_likeFill'gradients_1/pi/pow_grad/ones_like/Shape'gradients_1/pi/pow_grad/ones_like/Const*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ş
gradients_1/pi/pow_grad/SelectSelectgradients_1/pi/pow_grad/Greater
pi/truediv!gradients_1/pi/pow_grad/ones_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
gradients_1/pi/pow_grad/LogLoggradients_1/pi/pow_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
m
"gradients_1/pi/pow_grad/zeros_like	ZerosLike
pi/truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ž
 gradients_1/pi/pow_grad/Select_1Selectgradients_1/pi/pow_grad/Greatergradients_1/pi/pow_grad/Log"gradients_1/pi/pow_grad/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pi/pow_grad/mul_2Mul2gradients_1/pi/add_2_grad/tuple/control_dependencypi/pow*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pi/pow_grad/mul_3Mulgradients_1/pi/pow_grad/mul_2 gradients_1/pi/pow_grad/Select_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients_1/pi/pow_grad/Sum_1Sumgradients_1/pi/pow_grad/mul_3/gradients_1/pi/pow_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0

!gradients_1/pi/pow_grad/Reshape_1Reshapegradients_1/pi/pow_grad/Sum_1gradients_1/pi/pow_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
v
(gradients_1/pi/pow_grad/tuple/group_depsNoOp ^gradients_1/pi/pow_grad/Reshape"^gradients_1/pi/pow_grad/Reshape_1
î
0gradients_1/pi/pow_grad/tuple/control_dependencyIdentitygradients_1/pi/pow_grad/Reshape)^gradients_1/pi/pow_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*2
_class(
&$loc:@gradients_1/pi/pow_grad/Reshape
ă
2gradients_1/pi/pow_grad/tuple/control_dependency_1Identity!gradients_1/pi/pow_grad/Reshape_1)^gradients_1/pi/pow_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/pi/pow_grad/Reshape_1*
_output_shapes
: *
T0
u
2gradients_1/pi/mul_1_grad/BroadcastGradientArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 
|
2gradients_1/pi/mul_1_grad/BroadcastGradientArgs/s1Const*
_output_shapes
:*
valueB:*
dtype0
í
/gradients_1/pi/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients_1/pi/mul_1_grad/BroadcastGradientArgs/s02gradients_1/pi/mul_1_grad/BroadcastGradientArgs/s1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients_1/pi/mul_1_grad/MulMul4gradients_1/pi/add_2_grad/tuple/control_dependency_1pi/log_std/read*
_output_shapes
:*
T0
y
/gradients_1/pi/mul_1_grad/Sum/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
˛
gradients_1/pi/mul_1_grad/SumSumgradients_1/pi/mul_1_grad/Mul/gradients_1/pi/mul_1_grad/Sum/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
j
'gradients_1/pi/mul_1_grad/Reshape/shapeConst*
dtype0*
valueB *
_output_shapes
: 
Ł
!gradients_1/pi/mul_1_grad/ReshapeReshapegradients_1/pi/mul_1_grad/Sum'gradients_1/pi/mul_1_grad/Reshape/shape*
_output_shapes
: *
T0*
Tshape0

gradients_1/pi/mul_1_grad/Mul_1Mul
pi/mul_1/x4gradients_1/pi/add_2_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
x
*gradients_1/pi/mul_1_grad/tuple/group_depsNoOp ^gradients_1/pi/mul_1_grad/Mul_1"^gradients_1/pi/mul_1_grad/Reshape
ĺ
2gradients_1/pi/mul_1_grad/tuple/control_dependencyIdentity!gradients_1/pi/mul_1_grad/Reshape+^gradients_1/pi/mul_1_grad/tuple/group_deps*
_output_shapes
: *
T0*4
_class*
(&loc:@gradients_1/pi/mul_1_grad/Reshape
ç
4gradients_1/pi/mul_1_grad/tuple/control_dependency_1Identitygradients_1/pi/mul_1_grad/Mul_1+^gradients_1/pi/mul_1_grad/tuple/group_deps*
_output_shapes
:*
T0*2
_class(
&$loc:@gradients_1/pi/mul_1_grad/Mul_1
g
!gradients_1/pi/truediv_grad/ShapeShapepi/sub*
T0*
out_type0*
_output_shapes
:
m
#gradients_1/pi/truediv_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
Ď
1gradients_1/pi/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients_1/pi/truediv_grad/Shape#gradients_1/pi/truediv_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

#gradients_1/pi/truediv_grad/RealDivRealDiv0gradients_1/pi/pow_grad/tuple/control_dependencypi/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
gradients_1/pi/truediv_grad/SumSum#gradients_1/pi/truediv_grad/RealDiv1gradients_1/pi/truediv_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
˛
#gradients_1/pi/truediv_grad/ReshapeReshapegradients_1/pi/truediv_grad/Sum!gradients_1/pi/truediv_grad/Shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
gradients_1/pi/truediv_grad/NegNegpi/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

%gradients_1/pi/truediv_grad/RealDiv_1RealDivgradients_1/pi/truediv_grad/Negpi/add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

%gradients_1/pi/truediv_grad/RealDiv_2RealDiv%gradients_1/pi/truediv_grad/RealDiv_1pi/add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ą
gradients_1/pi/truediv_grad/mulMul0gradients_1/pi/pow_grad/tuple/control_dependency%gradients_1/pi/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
!gradients_1/pi/truediv_grad/Sum_1Sumgradients_1/pi/truediv_grad/mul3gradients_1/pi/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
Ť
%gradients_1/pi/truediv_grad/Reshape_1Reshape!gradients_1/pi/truediv_grad/Sum_1#gradients_1/pi/truediv_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0

,gradients_1/pi/truediv_grad/tuple/group_depsNoOp$^gradients_1/pi/truediv_grad/Reshape&^gradients_1/pi/truediv_grad/Reshape_1
ţ
4gradients_1/pi/truediv_grad/tuple/control_dependencyIdentity#gradients_1/pi/truediv_grad/Reshape-^gradients_1/pi/truediv_grad/tuple/group_deps*6
_class,
*(loc:@gradients_1/pi/truediv_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
÷
6gradients_1/pi/truediv_grad/tuple/control_dependency_1Identity%gradients_1/pi/truediv_grad/Reshape_1-^gradients_1/pi/truediv_grad/tuple/group_deps*8
_class.
,*loc:@gradients_1/pi/truediv_grad/Reshape_1*
_output_shapes
:*
T0
j
gradients_1/pi/sub_grad/ShapeShapePlaceholder_1*
T0*
_output_shapes
:*
out_type0
q
gradients_1/pi/sub_grad/Shape_1Shapepi/dense_2/BiasAdd*
out_type0*
T0*
_output_shapes
:
Ă
-gradients_1/pi/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/sub_grad/Shapegradients_1/pi/sub_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ç
gradients_1/pi/sub_grad/SumSum4gradients_1/pi/truediv_grad/tuple/control_dependency-gradients_1/pi/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ś
gradients_1/pi/sub_grad/ReshapeReshapegradients_1/pi/sub_grad/Sumgradients_1/pi/sub_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pi/sub_grad/NegNeg4gradients_1/pi/truediv_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
gradients_1/pi/sub_grad/Sum_1Sumgradients_1/pi/sub_grad/Neg/gradients_1/pi/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
Ź
!gradients_1/pi/sub_grad/Reshape_1Reshapegradients_1/pi/sub_grad/Sum_1gradients_1/pi/sub_grad/Shape_1*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
(gradients_1/pi/sub_grad/tuple/group_depsNoOp ^gradients_1/pi/sub_grad/Reshape"^gradients_1/pi/sub_grad/Reshape_1
î
0gradients_1/pi/sub_grad/tuple/control_dependencyIdentitygradients_1/pi/sub_grad/Reshape)^gradients_1/pi/sub_grad/tuple/group_deps*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*2
_class(
&$loc:@gradients_1/pi/sub_grad/Reshape
ô
2gradients_1/pi/sub_grad/tuple/control_dependency_1Identity!gradients_1/pi/sub_grad/Reshape_1)^gradients_1/pi/sub_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/pi/sub_grad/Reshape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
/gradients_1/pi/add_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
Ë
gradients_1/pi/add_1_grad/SumSum6gradients_1/pi/truediv_grad/tuple/control_dependency_1/gradients_1/pi/add_1_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
j
'gradients_1/pi/add_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Ł
!gradients_1/pi/add_1_grad/ReshapeReshapegradients_1/pi/add_1_grad/Sum'gradients_1/pi/add_1_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
: 

*gradients_1/pi/add_1_grad/tuple/group_depsNoOp"^gradients_1/pi/add_1_grad/Reshape7^gradients_1/pi/truediv_grad/tuple/control_dependency_1

2gradients_1/pi/add_1_grad/tuple/control_dependencyIdentity6gradients_1/pi/truediv_grad/tuple/control_dependency_1+^gradients_1/pi/add_1_grad/tuple/group_deps*8
_class.
,*loc:@gradients_1/pi/truediv_grad/Reshape_1*
T0*
_output_shapes
:
ç
4gradients_1/pi/add_1_grad/tuple/control_dependency_1Identity!gradients_1/pi/add_1_grad/Reshape+^gradients_1/pi/add_1_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/pi/add_1_grad/Reshape*
_output_shapes
: *
T0
Ž
/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients_1/pi/sub_grad/tuple/control_dependency_1*
T0*
_output_shapes
:*
data_formatNHWC
Ł
4gradients_1/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp0^gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad3^gradients_1/pi/sub_grad/tuple/control_dependency_1

<gradients_1/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity2gradients_1/pi/sub_grad/tuple/control_dependency_15^gradients_1/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*4
_class*
(&loc:@gradients_1/pi/sub_grad/Reshape_1

>gradients_1/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_1/pi/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*B
_class8
64loc:@gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad

gradients_1/pi/Exp_1_grad/mulMul2gradients_1/pi/add_1_grad/tuple/control_dependencypi/Exp_1*
T0*
_output_shapes
:
â
)gradients_1/pi/dense_2/MatMul_grad/MatMulMatMul<gradients_1/pi/dense_2/BiasAdd_grad/tuple/control_dependencypi/dense_2/kernel/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
+gradients_1/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh<gradients_1/pi/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes
:	

3gradients_1/pi/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_1/pi/dense_2/MatMul_grad/MatMul,^gradients_1/pi/dense_2/MatMul_grad/MatMul_1

;gradients_1/pi/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_1/pi/dense_2/MatMul_grad/MatMul4^gradients_1/pi/dense_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*<
_class2
0.loc:@gradients_1/pi/dense_2/MatMul_grad/MatMul

=gradients_1/pi/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_1/pi/dense_2/MatMul_grad/MatMul_14^gradients_1/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	*>
_class4
20loc:@gradients_1/pi/dense_2/MatMul_grad/MatMul_1

gradients_1/AddN_2AddN3gradients_1/pi/add_10_grad/tuple/control_dependency4gradients_1/pi/mul_1_grad/tuple/control_dependency_1gradients_1/pi/Exp_1_grad/mul*
N*1
_class'
%#loc:@gradients_1/pi/Sum_3_grad/Tile*
T0*
_output_shapes
:
ś
)gradients_1/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh;gradients_1/pi/dense_2/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_1/pi/dense_1/Tanh_grad/TanhGrad*
T0*
_output_shapes	
:*
data_formatNHWC

4gradients_1/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_1/pi/dense_1/Tanh_grad/TanhGrad

<gradients_1/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_1/pi/dense_1/Tanh_grad/TanhGrad5^gradients_1/pi/dense_1/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@gradients_1/pi/dense_1/Tanh_grad/TanhGrad*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_1/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_1/pi/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*B
_class8
64loc:@gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad
â
)gradients_1/pi/dense_1/MatMul_grad/MatMulMatMul<gradients_1/pi/dense_1/BiasAdd_grad/tuple/control_dependencypi/dense_1/kernel/read*
transpose_a( *
T0*
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
+gradients_1/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh<gradients_1/pi/dense_1/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_b( *
transpose_a(*
T0

3gradients_1/pi/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_1/pi/dense_1/MatMul_grad/MatMul,^gradients_1/pi/dense_1/MatMul_grad/MatMul_1

;gradients_1/pi/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_1/pi/dense_1/MatMul_grad/MatMul4^gradients_1/pi/dense_1/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients_1/pi/dense_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

=gradients_1/pi/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_1/pi/dense_1/MatMul_grad/MatMul_14^gradients_1/pi/dense_1/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
*>
_class4
20loc:@gradients_1/pi/dense_1/MatMul_grad/MatMul_1
˛
'gradients_1/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh;gradients_1/pi/dense_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
-gradients_1/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_1/pi/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
T0*
_output_shapes	
:

2gradients_1/pi/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad(^gradients_1/pi/dense/Tanh_grad/TanhGrad

:gradients_1/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_1/pi/dense/Tanh_grad/TanhGrad3^gradients_1/pi/dense/BiasAdd_grad/tuple/group_deps*:
_class0
.,loc:@gradients_1/pi/dense/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

<gradients_1/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad3^gradients_1/pi/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*@
_class6
42loc:@gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad
Ű
'gradients_1/pi/dense/MatMul_grad/MatMulMatMul:gradients_1/pi/dense/BiasAdd_grad/tuple/control_dependencypi/dense/kernel/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<
Ě
)gradients_1/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_1/pi/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	<

1gradients_1/pi/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_1/pi/dense/MatMul_grad/MatMul*^gradients_1/pi/dense/MatMul_grad/MatMul_1

9gradients_1/pi/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_1/pi/dense/MatMul_grad/MatMul2^gradients_1/pi/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*:
_class0
.,loc:@gradients_1/pi/dense/MatMul_grad/MatMul*
T0

;gradients_1/pi/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_1/pi/dense/MatMul_grad/MatMul_12^gradients_1/pi/dense/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients_1/pi/dense/MatMul_grad/MatMul_1*
_output_shapes
:	<*
T0
b
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙

	Reshape_4Reshape;gradients_1/pi/dense/MatMul_grad/tuple/control_dependency_1Reshape_4/shape*
_output_shapes	
:x*
Tshape0*
T0
b
Reshape_5/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0

	Reshape_5Reshape<gradients_1/pi/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_5/shape*
Tshape0*
_output_shapes	
:*
T0
b
Reshape_6/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0

	Reshape_6Reshape=gradients_1/pi/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_6/shape*
_output_shapes

:*
Tshape0*
T0
b
Reshape_7/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

	Reshape_7Reshape>gradients_1/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_7/shape*
T0*
_output_shapes	
:*
Tshape0
b
Reshape_8/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0

	Reshape_8Reshape=gradients_1/pi/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_8/shape*
_output_shapes	
:*
T0*
Tshape0
b
Reshape_9/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0

	Reshape_9Reshape>gradients_1/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_9/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
n

Reshape_10Reshapegradients_1/AddN_2Reshape_10/shape*
T0*
_output_shapes
:*
Tshape0
c
Reshape_11/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y

Reshape_11Reshapegradients_1/Softplus_grad/mulReshape_11/shape*
Tshape0*
T0*
_output_shapes
:
O
concat_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
š
concat_2ConcatV2	Reshape_4	Reshape_5	Reshape_6	Reshape_7	Reshape_8	Reshape_9
Reshape_10
Reshape_11concat_2/axis*
N*
_output_shapes

:*

Tidx0*
T0
l
PyFunc_2PyFuncconcat_2*
Tout
2*
token
pyfunc_2*
Tin
2*
_output_shapes

:
p
Const_6Const*5
value,B*"  <                       *
_output_shapes
:*
dtype0
S
split_2/split_dimConst*
_output_shapes
: *
value	B : *
dtype0
Š
split_2SplitVPyFunc_2Const_6split_2/split_dim*
	num_split*

Tlen0*J
_output_shapes8
6:x:::::::*
T0
a
Reshape_12/shapeConst*
valueB"<      *
dtype0*
_output_shapes
:
h

Reshape_12Reshapesplit_2Reshape_12/shape*
Tshape0*
T0*
_output_shapes
:	<
[
Reshape_13/shapeConst*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_13Reshape	split_2:1Reshape_13/shape*
_output_shapes	
:*
T0*
Tshape0
a
Reshape_14/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
k

Reshape_14Reshape	split_2:2Reshape_14/shape*
T0* 
_output_shapes
:
*
Tshape0
[
Reshape_15/shapeConst*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_15Reshape	split_2:3Reshape_15/shape*
T0*
Tshape0*
_output_shapes	
:
a
Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
j

Reshape_16Reshape	split_2:4Reshape_16/shape*
_output_shapes
:	*
T0*
Tshape0
Z
Reshape_17/shapeConst*
_output_shapes
:*
dtype0*
valueB:
e

Reshape_17Reshape	split_2:5Reshape_17/shape*
_output_shapes
:*
Tshape0*
T0
Z
Reshape_18/shapeConst*
valueB:*
_output_shapes
:*
dtype0
e

Reshape_18Reshape	split_2:6Reshape_18/shape*
Tshape0*
_output_shapes
:*
T0
S
Reshape_19/shapeConst*
_output_shapes
: *
dtype0*
valueB 
a

Reshape_19Reshape	split_2:7Reshape_19/shape*
T0*
Tshape0*
_output_shapes
: 

beta1_power_1/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*
dtype0*(
_class
loc:@penalty/penalty_param

beta1_power_1
VariableV2*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
shape: *
shared_name *
dtype0*
	container 
ž
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
x
beta1_power_1/readIdentitybeta1_power_1*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param

beta2_power_1/initial_valueConst*
dtype0*(
_class
loc:@penalty/penalty_param*
valueB
 *wž?*
_output_shapes
: 

beta2_power_1
VariableV2*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
	container *
shared_name *
dtype0*
shape: 
ž
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(
x
beta2_power_1/readIdentitybeta2_power_1*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Ť
6pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@pi/dense/kernel*
dtype0*
valueB"<      *
_output_shapes
:

,pi/dense/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *"
_class
loc:@pi/dense/kernel*
valueB
 *    *
dtype0
ô
&pi/dense/kernel/Adam/Initializer/zerosFill6pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,pi/dense/kernel/Adam/Initializer/zeros/Const*"
_class
loc:@pi/dense/kernel*
T0*

index_type0*
_output_shapes
:	<
Ž
pi/dense/kernel/Adam
VariableV2*
shape:	<*
_output_shapes
:	<*
	container *
dtype0*"
_class
loc:@pi/dense/kernel*
shared_name 
Ú
pi/dense/kernel/Adam/AssignAssignpi/dense/kernel/Adam&pi/dense/kernel/Adam/Initializer/zeros*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
use_locking(

pi/dense/kernel/Adam/readIdentitypi/dense/kernel/Adam*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
T0
­
8pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*"
_class
loc:@pi/dense/kernel*
valueB"<      

.pi/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    *"
_class
loc:@pi/dense/kernel
ú
(pi/dense/kernel/Adam_1/Initializer/zerosFill8pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.pi/dense/kernel/Adam_1/Initializer/zeros/Const*

index_type0*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	<
°
pi/dense/kernel/Adam_1
VariableV2*
dtype0*
shape:	<*
shared_name *"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
	container 
ŕ
pi/dense/kernel/Adam_1/AssignAssignpi/dense/kernel/Adam_1(pi/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel

pi/dense/kernel/Adam_1/readIdentitypi/dense/kernel/Adam_1*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
T0

$pi/dense/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    * 
_class
loc:@pi/dense/bias*
_output_shapes	
:
˘
pi/dense/bias/Adam
VariableV2*
shared_name *
	container * 
_class
loc:@pi/dense/bias*
dtype0*
_output_shapes	
:*
shape:
Î
pi/dense/bias/Adam/AssignAssignpi/dense/bias/Adam$pi/dense/bias/Adam/Initializer/zeros*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
validate_shape(*
use_locking(

pi/dense/bias/Adam/readIdentitypi/dense/bias/Adam*
_output_shapes	
:*
T0* 
_class
loc:@pi/dense/bias

&pi/dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_output_shapes	
:*
dtype0* 
_class
loc:@pi/dense/bias
¤
pi/dense/bias/Adam_1
VariableV2*
shared_name *
shape:*
	container *
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
dtype0
Ô
pi/dense/bias/Adam_1/AssignAssignpi/dense/bias/Adam_1&pi/dense/bias/Adam_1/Initializer/zeros* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(

pi/dense/bias/Adam_1/readIdentitypi/dense/bias/Adam_1*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes	
:
Ż
8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *
dtype0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes
:

.pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*$
_class
loc:@pi/dense_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
ý
(pi/dense_1/kernel/Adam/Initializer/zerosFill8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.pi/dense_1/kernel/Adam/Initializer/zeros/Const*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:
*

index_type0
´
pi/dense_1/kernel/Adam
VariableV2*
	container *
shape:
* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
dtype0*
shared_name 
ă
pi/dense_1/kernel/Adam/AssignAssignpi/dense_1/kernel/Adam(pi/dense_1/kernel/Adam/Initializer/zeros*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(*
validate_shape(

pi/dense_1/kernel/Adam/readIdentitypi/dense_1/kernel/Adam*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

ą
:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@pi/dense_1/kernel*
_output_shapes
:*
valueB"      *
dtype0

0pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *$
_class
loc:@pi/dense_1/kernel*
dtype0

*pi/dense_1/kernel/Adam_1/Initializer/zerosFill:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0pi/dense_1/kernel/Adam_1/Initializer/zeros/Const*

index_type0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
T0
ś
pi/dense_1/kernel/Adam_1
VariableV2*
shared_name *
shape:
*
	container *
dtype0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel
é
pi/dense_1/kernel/Adam_1/AssignAssignpi/dense_1/kernel/Adam_1*pi/dense_1/kernel/Adam_1/Initializer/zeros*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(* 
_output_shapes
:


pi/dense_1/kernel/Adam_1/readIdentitypi/dense_1/kernel/Adam_1*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel

&pi/dense_1/bias/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias
Ś
pi/dense_1/bias/Adam
VariableV2*
shape:*
	container *"
_class
loc:@pi/dense_1/bias*
shared_name *
dtype0*
_output_shapes	
:
Ö
pi/dense_1/bias/Adam/AssignAssignpi/dense_1/bias/Adam&pi/dense_1/bias/Adam/Initializer/zeros*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes	
:

pi/dense_1/bias/Adam/readIdentitypi/dense_1/bias/Adam*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0

(pi/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *"
_class
loc:@pi/dense_1/bias
¨
pi/dense_1/bias/Adam_1
VariableV2*
	container *
dtype0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
shape:*
shared_name 
Ü
pi/dense_1/bias/Adam_1/AssignAssignpi/dense_1/bias/Adam_1(pi/dense_1/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(

pi/dense_1/bias/Adam_1/readIdentitypi/dense_1/bias/Adam_1*
T0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias
Ľ
(pi/dense_2/kernel/Adam/Initializer/zerosConst*$
_class
loc:@pi/dense_2/kernel*
dtype0*
valueB	*    *
_output_shapes
:	
˛
pi/dense_2/kernel/Adam
VariableV2*
_output_shapes
:	*
dtype0*
	container *
shape:	*$
_class
loc:@pi/dense_2/kernel*
shared_name 
â
pi/dense_2/kernel/Adam/AssignAssignpi/dense_2/kernel/Adam(pi/dense_2/kernel/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	

pi/dense_2/kernel/Adam/readIdentitypi/dense_2/kernel/Adam*
_output_shapes
:	*
T0*$
_class
loc:@pi/dense_2/kernel
§
*pi/dense_2/kernel/Adam_1/Initializer/zerosConst*$
_class
loc:@pi/dense_2/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
´
pi/dense_2/kernel/Adam_1
VariableV2*
shared_name *
_output_shapes
:	*
	container *$
_class
loc:@pi/dense_2/kernel*
shape:	*
dtype0
č
pi/dense_2/kernel/Adam_1/AssignAssignpi/dense_2/kernel/Adam_1*pi/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
T0*
use_locking(

pi/dense_2/kernel/Adam_1/readIdentitypi/dense_2/kernel/Adam_1*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
:	

&pi/dense_2/bias/Adam/Initializer/zerosConst*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
dtype0*
valueB*    
¤
pi/dense_2/bias/Adam
VariableV2*
_output_shapes
:*
shape:*"
_class
loc:@pi/dense_2/bias*
dtype0*
shared_name *
	container 
Ő
pi/dense_2/bias/Adam/AssignAssignpi/dense_2/bias/Adam&pi/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(

pi/dense_2/bias/Adam/readIdentitypi/dense_2/bias/Adam*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:

(pi/dense_2/bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
Ś
pi/dense_2/bias/Adam_1
VariableV2*
shape:*"
_class
loc:@pi/dense_2/bias*
shared_name *
dtype0*
	container *
_output_shapes
:
Ű
pi/dense_2/bias/Adam_1/AssignAssignpi/dense_2/bias/Adam_1(pi/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0

pi/dense_2/bias/Adam_1/readIdentitypi/dense_2/bias/Adam_1*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:

!pi/log_std/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@pi/log_std*
_output_shapes
:*
dtype0

pi/log_std/Adam
VariableV2*
dtype0*
	container *
shape:*
_output_shapes
:*
_class
loc:@pi/log_std*
shared_name 
Á
pi/log_std/Adam/AssignAssignpi/log_std/Adam!pi/log_std/Adam/Initializer/zeros*
use_locking(*
_class
loc:@pi/log_std*
_output_shapes
:*
T0*
validate_shape(
u
pi/log_std/Adam/readIdentitypi/log_std/Adam*
T0*
_class
loc:@pi/log_std*
_output_shapes
:

#pi/log_std/Adam_1/Initializer/zerosConst*
_output_shapes
:*
_class
loc:@pi/log_std*
dtype0*
valueB*    

pi/log_std/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@pi/log_std*
shared_name *
_output_shapes
:*
shape:
Ç
pi/log_std/Adam_1/AssignAssignpi/log_std/Adam_1#pi/log_std/Adam_1/Initializer/zeros*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*
_class
loc:@pi/log_std
y
pi/log_std/Adam_1/readIdentitypi/log_std/Adam_1*
_class
loc:@pi/log_std*
_output_shapes
:*
T0

.penalty/penalty_param/Adam_2/Initializer/zerosConst*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
valueB
 *    *
dtype0
Ş
penalty/penalty_param/Adam_2
VariableV2*
shape: *(
_class
loc:@penalty/penalty_param*
dtype0*
	container *
shared_name *
_output_shapes
: 
ď
#penalty/penalty_param/Adam_2/AssignAssignpenalty/penalty_param/Adam_2.penalty/penalty_param/Adam_2/Initializer/zeros*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0*
use_locking(

!penalty/penalty_param/Adam_2/readIdentitypenalty/penalty_param/Adam_2*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0

.penalty/penalty_param/Adam_3/Initializer/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: *(
_class
loc:@penalty/penalty_param
Ş
penalty/penalty_param/Adam_3
VariableV2*(
_class
loc:@penalty/penalty_param*
	container *
dtype0*
shape: *
_output_shapes
: *
shared_name 
ď
#penalty/penalty_param/Adam_3/AssignAssignpenalty/penalty_param/Adam_3.penalty/penalty_param/Adam_3/Initializer/zeros*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
T0

!penalty/penalty_param/Adam_3/readIdentitypenalty/penalty_param/Adam_3*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: 
Y
Adam_1/learning_rateConst*
_output_shapes
: *
valueB
 *RI9*
dtype0
Q
Adam_1/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
Q
Adam_1/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
Ţ
'Adam_1/update_pi/dense/kernel/ApplyAdam	ApplyAdampi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_12*
use_nesterov( *
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
use_locking( 
Đ
%Adam_1/update_pi/dense/bias/ApplyAdam	ApplyAdampi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_13*
use_nesterov( *
use_locking( * 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0
é
)Adam_1/update_pi/dense_1/kernel/ApplyAdam	ApplyAdampi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_14*
T0*
use_locking( *
use_nesterov( *$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

Ú
'Adam_1/update_pi/dense_1/bias/ApplyAdam	ApplyAdampi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_15*
use_locking( *
T0*
use_nesterov( *
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias
č
)Adam_1/update_pi/dense_2/kernel/ApplyAdam	ApplyAdampi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_16*
use_locking( *
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0*
use_nesterov( 
Ů
'Adam_1/update_pi/dense_2/bias/ApplyAdam	ApplyAdampi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_17*
_output_shapes
:*
T0*
use_locking( *
use_nesterov( *"
_class
loc:@pi/dense_2/bias
Ŕ
"Adam_1/update_pi/log_std/ApplyAdam	ApplyAdam
pi/log_stdpi/log_std/Adampi/log_std/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_18*
_class
loc:@pi/log_std*
use_locking( *
_output_shapes
:*
T0*
use_nesterov( 
ő
-Adam_1/update_penalty/penalty_param/ApplyAdam	ApplyAdampenalty/penalty_parampenalty/penalty_param/Adam_2penalty/penalty_param/Adam_3beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_19*
T0*
use_locking( *(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_nesterov( 
Ń

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1.^Adam_1/update_penalty/penalty_param/ApplyAdam&^Adam_1/update_pi/dense/bias/ApplyAdam(^Adam_1/update_pi/dense/kernel/ApplyAdam(^Adam_1/update_pi/dense_1/bias/ApplyAdam*^Adam_1/update_pi/dense_1/kernel/ApplyAdam(^Adam_1/update_pi/dense_2/bias/ApplyAdam*^Adam_1/update_pi/dense_2/kernel/ApplyAdam#^Adam_1/update_pi/log_std/ApplyAdam*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param
Ś
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
use_locking( *
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Ó
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2.^Adam_1/update_penalty/penalty_param/ApplyAdam&^Adam_1/update_pi/dense/bias/ApplyAdam(^Adam_1/update_pi/dense/kernel/ApplyAdam(^Adam_1/update_pi/dense_1/bias/ApplyAdam*^Adam_1/update_pi/dense_1/kernel/ApplyAdam(^Adam_1/update_pi/dense_2/bias/ApplyAdam*^Adam_1/update_pi/dense_2/kernel/ApplyAdam#^Adam_1/update_pi/log_std/ApplyAdam*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0
Ş
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0

Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1.^Adam_1/update_penalty/penalty_param/ApplyAdam&^Adam_1/update_pi/dense/bias/ApplyAdam(^Adam_1/update_pi/dense/kernel/ApplyAdam(^Adam_1/update_pi/dense_1/bias/ApplyAdam*^Adam_1/update_pi/dense_1/kernel/ApplyAdam(^Adam_1/update_pi/dense_2/bias/ApplyAdam*^Adam_1/update_pi/dense_2/kernel/ApplyAdam#^Adam_1/update_pi/log_std/ApplyAdam
l
Reshape_20/shapeConst^Adam_1*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
q

Reshape_20Reshapepi/dense/kernel/readReshape_20/shape*
_output_shapes	
:x*
T0*
Tshape0
l
Reshape_21/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
o

Reshape_21Reshapepi/dense/bias/readReshape_21/shape*
T0*
_output_shapes	
:*
Tshape0
l
Reshape_22/shapeConst^Adam_1*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
t

Reshape_22Reshapepi/dense_1/kernel/readReshape_22/shape*
Tshape0*
_output_shapes

:*
T0
l
Reshape_23/shapeConst^Adam_1*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
q

Reshape_23Reshapepi/dense_1/bias/readReshape_23/shape*
Tshape0*
T0*
_output_shapes	
:
l
Reshape_24/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
s

Reshape_24Reshapepi/dense_2/kernel/readReshape_24/shape*
T0*
Tshape0*
_output_shapes	
:
l
Reshape_25/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
p

Reshape_25Reshapepi/dense_2/bias/readReshape_25/shape*
_output_shapes
:*
T0*
Tshape0
l
Reshape_26/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
k

Reshape_26Reshapepi/log_std/readReshape_26/shape*
Tshape0*
T0*
_output_shapes
:
l
Reshape_27/shapeConst^Adam_1*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
v

Reshape_27Reshapepenalty/penalty_param/readReshape_27/shape*
_output_shapes
:*
T0*
Tshape0
X
concat_3/axisConst^Adam_1*
dtype0*
_output_shapes
: *
value	B : 
ż
concat_3ConcatV2
Reshape_20
Reshape_21
Reshape_22
Reshape_23
Reshape_24
Reshape_25
Reshape_26
Reshape_27concat_3/axis*
N*

Tidx0*
T0*
_output_shapes

:
h
PyFunc_3PyFuncconcat_3*
token
pyfunc_3*
Tout
2*
Tin
2*
_output_shapes
:
y
Const_7Const^Adam_1*5
value,B*"  <                       *
dtype0*
_output_shapes
:
\
split_3/split_dimConst^Adam_1*
value	B : *
dtype0*
_output_shapes
: 

split_3SplitVPyFunc_3Const_7split_3/split_dim*
	num_split*

Tlen0*
T0*4
_output_shapes"
 ::::::::
j
Reshape_28/shapeConst^Adam_1*
_output_shapes
:*
valueB"<      *
dtype0
h

Reshape_28Reshapesplit_3Reshape_28/shape*
Tshape0*
T0*
_output_shapes
:	<
d
Reshape_29/shapeConst^Adam_1*
dtype0*
valueB:*
_output_shapes
:
f

Reshape_29Reshape	split_3:1Reshape_29/shape*
_output_shapes	
:*
Tshape0*
T0
j
Reshape_30/shapeConst^Adam_1*
valueB"      *
_output_shapes
:*
dtype0
k

Reshape_30Reshape	split_3:2Reshape_30/shape*
Tshape0*
T0* 
_output_shapes
:

d
Reshape_31/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB:
f

Reshape_31Reshape	split_3:3Reshape_31/shape*
_output_shapes	
:*
T0*
Tshape0
j
Reshape_32/shapeConst^Adam_1*
valueB"      *
_output_shapes
:*
dtype0
j

Reshape_32Reshape	split_3:4Reshape_32/shape*
T0*
Tshape0*
_output_shapes
:	
c
Reshape_33/shapeConst^Adam_1*
dtype0*
valueB:*
_output_shapes
:
e

Reshape_33Reshape	split_3:5Reshape_33/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_34/shapeConst^Adam_1*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_34Reshape	split_3:6Reshape_34/shape*
T0*
_output_shapes
:*
Tshape0
\
Reshape_35/shapeConst^Adam_1*
valueB *
dtype0*
_output_shapes
: 
a

Reshape_35Reshape	split_3:7Reshape_35/shape*
_output_shapes
: *
T0*
Tshape0
Ś
Assign_1Assignpi/dense/kernel
Reshape_28*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0*
validate_shape(

Assign_2Assignpi/dense/bias
Reshape_29*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias
Ť
Assign_3Assignpi/dense_1/kernel
Reshape_30*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
use_locking(
˘
Assign_4Assignpi/dense_1/bias
Reshape_31*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0*
validate_shape(
Ş
Assign_5Assignpi/dense_2/kernel
Reshape_32*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	*
use_locking(
Ą
Assign_6Assignpi/dense_2/bias
Reshape_33*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:

Assign_7Assign
pi/log_std
Reshape_34*
use_locking(*
_output_shapes
:*
_class
loc:@pi/log_std*
T0*
validate_shape(
Š
Assign_8Assignpenalty/penalty_param
Reshape_35*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0
u
group_deps_2NoOp^Adam_1	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8
,
group_deps_3NoOp^Adam_1^group_deps_2
U
sub_3SubPlaceholder_4
vf/Squeeze*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
J
pow/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0
F
powPowsub_3pow/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_8Const*
valueB: *
_output_shapes
:*
dtype0
Z
Mean_3MeanpowConst_8*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
U
sub_4SubPlaceholder_5
vc/Squeeze*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
L
pow_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
J
pow_1Powsub_4pow_1/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_9Const*
dtype0*
_output_shapes
:*
valueB: 
\
Mean_4Meanpow_1Const_9*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
?
add_2AddV2Mean_3Mean_4*
T0*
_output_shapes
: 
T
gradients_2/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Z
gradients_2/grad_ys_0Const*
valueB
 *  ?*
_output_shapes
: *
dtype0
u
gradients_2/FillFillgradients_2/Shapegradients_2/grad_ys_0*
_output_shapes
: *
T0*

index_type0
B
'gradients_2/add_2_grad/tuple/group_depsNoOp^gradients_2/Fill
˝
/gradients_2/add_2_grad/tuple/control_dependencyIdentitygradients_2/Fill(^gradients_2/add_2_grad/tuple/group_deps*
T0*
_output_shapes
: *#
_class
loc:@gradients_2/Fill
ż
1gradients_2/add_2_grad/tuple/control_dependency_1Identitygradients_2/Fill(^gradients_2/add_2_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_2/Fill*
_output_shapes
: 
o
%gradients_2/Mean_3_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
ľ
gradients_2/Mean_3_grad/ReshapeReshape/gradients_2/add_2_grad/tuple/control_dependency%gradients_2/Mean_3_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
`
gradients_2/Mean_3_grad/ShapeShapepow*
out_type0*
T0*
_output_shapes
:
¤
gradients_2/Mean_3_grad/TileTilegradients_2/Mean_3_grad/Reshapegradients_2/Mean_3_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients_2/Mean_3_grad/Shape_1Shapepow*
_output_shapes
:*
out_type0*
T0
b
gradients_2/Mean_3_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_2/Mean_3_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
˘
gradients_2/Mean_3_grad/ProdProdgradients_2/Mean_3_grad/Shape_1gradients_2/Mean_3_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
i
gradients_2/Mean_3_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ś
gradients_2/Mean_3_grad/Prod_1Prodgradients_2/Mean_3_grad/Shape_2gradients_2/Mean_3_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
c
!gradients_2/Mean_3_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0

gradients_2/Mean_3_grad/MaximumMaximumgradients_2/Mean_3_grad/Prod_1!gradients_2/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_2/Mean_3_grad/floordivFloorDivgradients_2/Mean_3_grad/Prodgradients_2/Mean_3_grad/Maximum*
T0*
_output_shapes
: 

gradients_2/Mean_3_grad/CastCast gradients_2/Mean_3_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0*
Truncate( 

gradients_2/Mean_3_grad/truedivRealDivgradients_2/Mean_3_grad/Tilegradients_2/Mean_3_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
%gradients_2/Mean_4_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
ˇ
gradients_2/Mean_4_grad/ReshapeReshape1gradients_2/add_2_grad/tuple/control_dependency_1%gradients_2/Mean_4_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
b
gradients_2/Mean_4_grad/ShapeShapepow_1*
_output_shapes
:*
T0*
out_type0
¤
gradients_2/Mean_4_grad/TileTilegradients_2/Mean_4_grad/Reshapegradients_2/Mean_4_grad/Shape*

Tmultiples0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
d
gradients_2/Mean_4_grad/Shape_1Shapepow_1*
_output_shapes
:*
T0*
out_type0
b
gradients_2/Mean_4_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
g
gradients_2/Mean_4_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
˘
gradients_2/Mean_4_grad/ProdProdgradients_2/Mean_4_grad/Shape_1gradients_2/Mean_4_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
i
gradients_2/Mean_4_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
Ś
gradients_2/Mean_4_grad/Prod_1Prodgradients_2/Mean_4_grad/Shape_2gradients_2/Mean_4_grad/Const_1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
c
!gradients_2/Mean_4_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0

gradients_2/Mean_4_grad/MaximumMaximumgradients_2/Mean_4_grad/Prod_1!gradients_2/Mean_4_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_2/Mean_4_grad/floordivFloorDivgradients_2/Mean_4_grad/Prodgradients_2/Mean_4_grad/Maximum*
_output_shapes
: *
T0

gradients_2/Mean_4_grad/CastCast gradients_2/Mean_4_grad/floordiv*

DstT0*
Truncate( *

SrcT0*
_output_shapes
: 

gradients_2/Mean_4_grad/truedivRealDivgradients_2/Mean_4_grad/Tilegradients_2/Mean_4_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients_2/pow_grad/ShapeShapesub_3*
T0*
out_type0*
_output_shapes
:
_
gradients_2/pow_grad/Shape_1Shapepow/y*
T0*
_output_shapes
: *
out_type0
ş
*gradients_2/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/pow_grad/Shapegradients_2/pow_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
u
gradients_2/pow_grad/mulMulgradients_2/Mean_3_grad/truedivpow/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients_2/pow_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
c
gradients_2/pow_grad/subSubpow/ygradients_2/pow_grad/sub/y*
_output_shapes
: *
T0
n
gradients_2/pow_grad/PowPowsub_3gradients_2/pow_grad/sub*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_2/pow_grad/mul_1Mulgradients_2/pow_grad/mulgradients_2/pow_grad/Pow*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
gradients_2/pow_grad/SumSumgradients_2/pow_grad/mul_1*gradients_2/pow_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_2/pow_grad/ReshapeReshapegradients_2/pow_grad/Sumgradients_2/pow_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
gradients_2/pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients_2/pow_grad/GreaterGreatersub_3gradients_2/pow_grad/Greater/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
i
$gradients_2/pow_grad/ones_like/ShapeShapesub_3*
out_type0*
_output_shapes
:*
T0
i
$gradients_2/pow_grad/ones_like/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
˛
gradients_2/pow_grad/ones_likeFill$gradients_2/pow_grad/ones_like/Shape$gradients_2/pow_grad/ones_like/Const*
T0*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_2/pow_grad/SelectSelectgradients_2/pow_grad/Greatersub_3gradients_2/pow_grad/ones_like*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
j
gradients_2/pow_grad/LogLoggradients_2/pow_grad/Select*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
a
gradients_2/pow_grad/zeros_like	ZerosLikesub_3*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
gradients_2/pow_grad/Select_1Selectgradients_2/pow_grad/Greatergradients_2/pow_grad/Loggradients_2/pow_grad/zeros_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
gradients_2/pow_grad/mul_2Mulgradients_2/Mean_3_grad/truedivpow*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_2/pow_grad/mul_3Mulgradients_2/pow_grad/mul_2gradients_2/pow_grad/Select_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ť
gradients_2/pow_grad/Sum_1Sumgradients_2/pow_grad/mul_3,gradients_2/pow_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients_2/pow_grad/Reshape_1Reshapegradients_2/pow_grad/Sum_1gradients_2/pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients_2/pow_grad/tuple/group_depsNoOp^gradients_2/pow_grad/Reshape^gradients_2/pow_grad/Reshape_1
Ţ
-gradients_2/pow_grad/tuple/control_dependencyIdentitygradients_2/pow_grad/Reshape&^gradients_2/pow_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_2/pow_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
/gradients_2/pow_grad/tuple/control_dependency_1Identitygradients_2/pow_grad/Reshape_1&^gradients_2/pow_grad/tuple/group_deps*
T0*
_output_shapes
: *1
_class'
%#loc:@gradients_2/pow_grad/Reshape_1
a
gradients_2/pow_1_grad/ShapeShapesub_4*
out_type0*
T0*
_output_shapes
:
c
gradients_2/pow_1_grad/Shape_1Shapepow_1/y*
_output_shapes
: *
T0*
out_type0
Ŕ
,gradients_2/pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/pow_1_grad/Shapegradients_2/pow_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
y
gradients_2/pow_1_grad/mulMulgradients_2/Mean_4_grad/truedivpow_1/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
a
gradients_2/pow_1_grad/sub/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
i
gradients_2/pow_1_grad/subSubpow_1/ygradients_2/pow_1_grad/sub/y*
T0*
_output_shapes
: 
r
gradients_2/pow_1_grad/PowPowsub_4gradients_2/pow_1_grad/sub*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_2/pow_1_grad/mul_1Mulgradients_2/pow_1_grad/mulgradients_2/pow_1_grad/Pow*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
­
gradients_2/pow_1_grad/SumSumgradients_2/pow_1_grad/mul_1,gradients_2/pow_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:

gradients_2/pow_1_grad/ReshapeReshapegradients_2/pow_1_grad/Sumgradients_2/pow_1_grad/Shape*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
e
 gradients_2/pow_1_grad/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients_2/pow_1_grad/GreaterGreatersub_4 gradients_2/pow_1_grad/Greater/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
&gradients_2/pow_1_grad/ones_like/ShapeShapesub_4*
T0*
_output_shapes
:*
out_type0
k
&gradients_2/pow_1_grad/ones_like/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
¸
 gradients_2/pow_1_grad/ones_likeFill&gradients_2/pow_1_grad/ones_like/Shape&gradients_2/pow_1_grad/ones_like/Const*
T0*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_2/pow_1_grad/SelectSelectgradients_2/pow_1_grad/Greatersub_4 gradients_2/pow_1_grad/ones_like*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
n
gradients_2/pow_1_grad/LogLoggradients_2/pow_1_grad/Select*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
!gradients_2/pow_1_grad/zeros_like	ZerosLikesub_4*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ś
gradients_2/pow_1_grad/Select_1Selectgradients_2/pow_1_grad/Greatergradients_2/pow_1_grad/Log!gradients_2/pow_1_grad/zeros_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
gradients_2/pow_1_grad/mul_2Mulgradients_2/Mean_4_grad/truedivpow_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_2/pow_1_grad/mul_3Mulgradients_2/pow_1_grad/mul_2gradients_2/pow_1_grad/Select_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
gradients_2/pow_1_grad/Sum_1Sumgradients_2/pow_1_grad/mul_3.gradients_2/pow_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0

 gradients_2/pow_1_grad/Reshape_1Reshapegradients_2/pow_1_grad/Sum_1gradients_2/pow_1_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
s
'gradients_2/pow_1_grad/tuple/group_depsNoOp^gradients_2/pow_1_grad/Reshape!^gradients_2/pow_1_grad/Reshape_1
ć
/gradients_2/pow_1_grad/tuple/control_dependencyIdentitygradients_2/pow_1_grad/Reshape(^gradients_2/pow_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_2/pow_1_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
1gradients_2/pow_1_grad/tuple/control_dependency_1Identity gradients_2/pow_1_grad/Reshape_1(^gradients_2/pow_1_grad/tuple/group_deps*
_output_shapes
: *3
_class)
'%loc:@gradients_2/pow_1_grad/Reshape_1*
T0
i
gradients_2/sub_3_grad/ShapeShapePlaceholder_4*
T0*
out_type0*
_output_shapes
:
h
gradients_2/sub_3_grad/Shape_1Shape
vf/Squeeze*
out_type0*
T0*
_output_shapes
:
Ŕ
,gradients_2/sub_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/sub_3_grad/Shapegradients_2/sub_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ž
gradients_2/sub_3_grad/SumSum-gradients_2/pow_grad/tuple/control_dependency,gradients_2/sub_3_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients_2/sub_3_grad/ReshapeReshapegradients_2/sub_3_grad/Sumgradients_2/sub_3_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
~
gradients_2/sub_3_grad/NegNeg-gradients_2/pow_grad/tuple/control_dependency*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ż
gradients_2/sub_3_grad/Sum_1Sumgradients_2/sub_3_grad/Neg.gradients_2/sub_3_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
Ľ
 gradients_2/sub_3_grad/Reshape_1Reshapegradients_2/sub_3_grad/Sum_1gradients_2/sub_3_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_2/sub_3_grad/tuple/group_depsNoOp^gradients_2/sub_3_grad/Reshape!^gradients_2/sub_3_grad/Reshape_1
ć
/gradients_2/sub_3_grad/tuple/control_dependencyIdentitygradients_2/sub_3_grad/Reshape(^gradients_2/sub_3_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*1
_class'
%#loc:@gradients_2/sub_3_grad/Reshape*
T0
ě
1gradients_2/sub_3_grad/tuple/control_dependency_1Identity gradients_2/sub_3_grad/Reshape_1(^gradients_2/sub_3_grad/tuple/group_deps*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*3
_class)
'%loc:@gradients_2/sub_3_grad/Reshape_1
i
gradients_2/sub_4_grad/ShapeShapePlaceholder_5*
_output_shapes
:*
out_type0*
T0
h
gradients_2/sub_4_grad/Shape_1Shape
vc/Squeeze*
out_type0*
_output_shapes
:*
T0
Ŕ
,gradients_2/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/sub_4_grad/Shapegradients_2/sub_4_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ŕ
gradients_2/sub_4_grad/SumSum/gradients_2/pow_1_grad/tuple/control_dependency,gradients_2/sub_4_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients_2/sub_4_grad/ReshapeReshapegradients_2/sub_4_grad/Sumgradients_2/sub_4_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0

gradients_2/sub_4_grad/NegNeg/gradients_2/pow_1_grad/tuple/control_dependency*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
gradients_2/sub_4_grad/Sum_1Sumgradients_2/sub_4_grad/Neg.gradients_2/sub_4_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ľ
 gradients_2/sub_4_grad/Reshape_1Reshapegradients_2/sub_4_grad/Sum_1gradients_2/sub_4_grad/Shape_1*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
'gradients_2/sub_4_grad/tuple/group_depsNoOp^gradients_2/sub_4_grad/Reshape!^gradients_2/sub_4_grad/Reshape_1
ć
/gradients_2/sub_4_grad/tuple/control_dependencyIdentitygradients_2/sub_4_grad/Reshape(^gradients_2/sub_4_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*1
_class'
%#loc:@gradients_2/sub_4_grad/Reshape
ě
1gradients_2/sub_4_grad/tuple/control_dependency_1Identity gradients_2/sub_4_grad/Reshape_1(^gradients_2/sub_4_grad/tuple/group_deps*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*3
_class)
'%loc:@gradients_2/sub_4_grad/Reshape_1
s
!gradients_2/vf/Squeeze_grad/ShapeShapevf/dense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
Ä
#gradients_2/vf/Squeeze_grad/ReshapeReshape1gradients_2/sub_3_grad/tuple/control_dependency_1!gradients_2/vf/Squeeze_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
s
!gradients_2/vc/Squeeze_grad/ShapeShapevc/dense_2/BiasAdd*
T0*
_output_shapes
:*
out_type0
Ä
#gradients_2/vc/Squeeze_grad/ReshapeReshape1gradients_2/sub_4_grad/tuple/control_dependency_1!gradients_2/vc/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_2/vf/Squeeze_grad/Reshape*
data_formatNHWC*
T0*
_output_shapes
:

4gradients_2/vf/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_2/vf/Squeeze_grad/Reshape0^gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGrad

<gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_2/vf/Squeeze_grad/Reshape5^gradients_2/vf/dense_2/BiasAdd_grad/tuple/group_deps*6
_class,
*(loc:@gradients_2/vf/Squeeze_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

>gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_2/vf/dense_2/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0

/gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_2/vc/Squeeze_grad/Reshape*
T0*
_output_shapes
:*
data_formatNHWC

4gradients_2/vc/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_2/vc/Squeeze_grad/Reshape0^gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGrad

<gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_2/vc/Squeeze_grad/Reshape5^gradients_2/vc/dense_2/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*6
_class,
*(loc:@gradients_2/vc/Squeeze_grad/Reshape

>gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_2/vc/dense_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*B
_class8
64loc:@gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGrad
â
)gradients_2/vf/dense_2/MatMul_grad/MatMulMatMul<gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependencyvf/dense_2/kernel/read*
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0
Ô
+gradients_2/vf/dense_2/MatMul_grad/MatMul_1MatMulvf/dense_1/Tanh<gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 

3gradients_2/vf/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vf/dense_2/MatMul_grad/MatMul,^gradients_2/vf/dense_2/MatMul_grad/MatMul_1

;gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vf/dense_2/MatMul_grad/MatMul4^gradients_2/vf/dense_2/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*<
_class2
0.loc:@gradients_2/vf/dense_2/MatMul_grad/MatMul

=gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vf/dense_2/MatMul_grad/MatMul_14^gradients_2/vf/dense_2/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_2/vf/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
â
)gradients_2/vc/dense_2/MatMul_grad/MatMulMatMul<gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependencyvc/dense_2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( *
transpose_b(
Ô
+gradients_2/vc/dense_2/MatMul_grad/MatMul_1MatMulvc/dense_1/Tanh<gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	*
T0*
transpose_b( 

3gradients_2/vc/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vc/dense_2/MatMul_grad/MatMul,^gradients_2/vc/dense_2/MatMul_grad/MatMul_1

;gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vc/dense_2/MatMul_grad/MatMul4^gradients_2/vc/dense_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*<
_class2
0.loc:@gradients_2/vc/dense_2/MatMul_grad/MatMul

=gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vc/dense_2/MatMul_grad/MatMul_14^gradients_2/vc/dense_2/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_2/vc/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	
ś
)gradients_2/vf/dense_1/Tanh_grad/TanhGradTanhGradvf/dense_1/Tanh;gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ś
)gradients_2/vc/dense_1/Tanh_grad/TanhGradTanhGradvc/dense_1/Tanh;gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
/gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_2/vf/dense_1/Tanh_grad/TanhGrad*
T0*
_output_shapes	
:*
data_formatNHWC

4gradients_2/vf/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_2/vf/dense_1/Tanh_grad/TanhGrad

<gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_2/vf/dense_1/Tanh_grad/TanhGrad5^gradients_2/vf/dense_1/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@gradients_2/vf/dense_1/Tanh_grad/TanhGrad*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_2/vf/dense_1/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
Ś
/gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_2/vc/dense_1/Tanh_grad/TanhGrad*
T0*
_output_shapes	
:*
data_formatNHWC

4gradients_2/vc/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_2/vc/dense_1/Tanh_grad/TanhGrad

<gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_2/vc/dense_1/Tanh_grad/TanhGrad5^gradients_2/vc/dense_1/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*<
_class2
0.loc:@gradients_2/vc/dense_1/Tanh_grad/TanhGrad*
T0

>gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_2/vc/dense_1/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
â
)gradients_2/vf/dense_1/MatMul_grad/MatMulMatMul<gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependencyvf/dense_1/kernel/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
+gradients_2/vf/dense_1/MatMul_grad/MatMul_1MatMulvf/dense/Tanh<gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( * 
_output_shapes
:
*
T0*
transpose_a(

3gradients_2/vf/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vf/dense_1/MatMul_grad/MatMul,^gradients_2/vf/dense_1/MatMul_grad/MatMul_1

;gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vf/dense_1/MatMul_grad/MatMul4^gradients_2/vf/dense_1/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/vf/dense_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vf/dense_1/MatMul_grad/MatMul_14^gradients_2/vf/dense_1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*>
_class4
20loc:@gradients_2/vf/dense_1/MatMul_grad/MatMul_1*
T0
â
)gradients_2/vc/dense_1/MatMul_grad/MatMulMatMul<gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependencyvc/dense_1/kernel/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
+gradients_2/vc/dense_1/MatMul_grad/MatMul_1MatMulvc/dense/Tanh<gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:


3gradients_2/vc/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vc/dense_1/MatMul_grad/MatMul,^gradients_2/vc/dense_1/MatMul_grad/MatMul_1

;gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vc/dense_1/MatMul_grad/MatMul4^gradients_2/vc/dense_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*<
_class2
0.loc:@gradients_2/vc/dense_1/MatMul_grad/MatMul

=gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vc/dense_1/MatMul_grad/MatMul_14^gradients_2/vc/dense_1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*>
_class4
20loc:@gradients_2/vc/dense_1/MatMul_grad/MatMul_1
˛
'gradients_2/vf/dense/Tanh_grad/TanhGradTanhGradvf/dense/Tanh;gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˛
'gradients_2/vc/dense/Tanh_grad/TanhGradTanhGradvc/dense/Tanh;gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
-gradients_2/vf/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_2/vf/dense/Tanh_grad/TanhGrad*
T0*
_output_shapes	
:*
data_formatNHWC

2gradients_2/vf/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_2/vf/dense/BiasAdd_grad/BiasAddGrad(^gradients_2/vf/dense/Tanh_grad/TanhGrad

:gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_2/vf/dense/Tanh_grad/TanhGrad3^gradients_2/vf/dense/BiasAdd_grad/tuple/group_deps*:
_class0
.,loc:@gradients_2/vf/dense/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

<gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_2/vf/dense/BiasAdd_grad/BiasAddGrad3^gradients_2/vf/dense/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@gradients_2/vf/dense/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
˘
-gradients_2/vc/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_2/vc/dense/Tanh_grad/TanhGrad*
_output_shapes	
:*
data_formatNHWC*
T0

2gradients_2/vc/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_2/vc/dense/BiasAdd_grad/BiasAddGrad(^gradients_2/vc/dense/Tanh_grad/TanhGrad

:gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_2/vc/dense/Tanh_grad/TanhGrad3^gradients_2/vc/dense/BiasAdd_grad/tuple/group_deps*:
_class0
.,loc:@gradients_2/vc/dense/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

<gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_2/vc/dense/BiasAdd_grad/BiasAddGrad3^gradients_2/vc/dense/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@gradients_2/vc/dense/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
Ű
'gradients_2/vf/dense/MatMul_grad/MatMulMatMul:gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependencyvf/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
transpose_b(
Ě
)gradients_2/vf/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	<*
transpose_a(*
transpose_b( *
T0

1gradients_2/vf/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_2/vf/dense/MatMul_grad/MatMul*^gradients_2/vf/dense/MatMul_grad/MatMul_1

9gradients_2/vf/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_2/vf/dense/MatMul_grad/MatMul2^gradients_2/vf/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
T0*:
_class0
.,loc:@gradients_2/vf/dense/MatMul_grad/MatMul

;gradients_2/vf/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_2/vf/dense/MatMul_grad/MatMul_12^gradients_2/vf/dense/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	<*<
_class2
0.loc:@gradients_2/vf/dense/MatMul_grad/MatMul_1
Ű
'gradients_2/vc/dense/MatMul_grad/MatMulMatMul:gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependencyvc/dense/kernel/read*
transpose_a( *
transpose_b(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
T0
Ě
)gradients_2/vc/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes
:	<*
T0*
transpose_a(

1gradients_2/vc/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_2/vc/dense/MatMul_grad/MatMul*^gradients_2/vc/dense/MatMul_grad/MatMul_1

9gradients_2/vc/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_2/vc/dense/MatMul_grad/MatMul2^gradients_2/vc/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
T0*:
_class0
.,loc:@gradients_2/vc/dense/MatMul_grad/MatMul

;gradients_2/vc/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_2/vc/dense/MatMul_grad/MatMul_12^gradients_2/vc/dense/MatMul_grad/tuple/group_deps*
_output_shapes
:	<*<
_class2
0.loc:@gradients_2/vc/dense/MatMul_grad/MatMul_1*
T0
c
Reshape_36/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_36Reshape;gradients_2/vf/dense/MatMul_grad/tuple/control_dependency_1Reshape_36/shape*
_output_shapes	
:x*
T0*
Tshape0
c
Reshape_37/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_37Reshape<gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_37/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_38/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:


Reshape_38Reshape=gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_38/shape*
_output_shapes

:*
T0*
Tshape0
c
Reshape_39/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:


Reshape_39Reshape>gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_39/shape*
_output_shapes	
:*
T0*
Tshape0
c
Reshape_40/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0


Reshape_40Reshape=gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_40/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_41/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_41Reshape>gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_41/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_42/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:


Reshape_42Reshape;gradients_2/vc/dense/MatMul_grad/tuple/control_dependency_1Reshape_42/shape*
T0*
Tshape0*
_output_shapes	
:x
c
Reshape_43/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0


Reshape_43Reshape<gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_43/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_44/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_44Reshape=gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_44/shape*
Tshape0*
T0*
_output_shapes

:
c
Reshape_45/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0


Reshape_45Reshape>gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_45/shape*
_output_shapes	
:*
T0*
Tshape0
c
Reshape_46/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_46Reshape=gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_46/shape*
Tshape0*
T0*
_output_shapes	
:
c
Reshape_47/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_47Reshape>gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_47/shape*
T0*
_output_shapes
:*
Tshape0
O
concat_4/axisConst*
_output_shapes
: *
value	B : *
dtype0
ď
concat_4ConcatV2
Reshape_36
Reshape_37
Reshape_38
Reshape_39
Reshape_40
Reshape_41
Reshape_42
Reshape_43
Reshape_44
Reshape_45
Reshape_46
Reshape_47concat_4/axis*
_output_shapes

:ü	*
T0*
N*

Tidx0
l
PyFunc_4PyFuncconcat_4*
Tout
2*
_output_shapes

:ü	*
Tin
2*
token
pyfunc_4

Const_10Const*E
value<B:"0 <                  <                 *
_output_shapes
:*
dtype0
S
split_4/split_dimConst*
value	B : *
_output_shapes
: *
dtype0
Č
split_4SplitVPyFunc_4Const_10split_4/split_dim*
T0*
	num_split*

Tlen0*h
_output_shapesV
T:x::::::x:::::
a
Reshape_48/shapeConst*
dtype0*
_output_shapes
:*
valueB"<      
h

Reshape_48Reshapesplit_4Reshape_48/shape*
Tshape0*
_output_shapes
:	<*
T0
[
Reshape_49/shapeConst*
dtype0*
_output_shapes
:*
valueB:
f

Reshape_49Reshape	split_4:1Reshape_49/shape*
_output_shapes	
:*
Tshape0*
T0
a
Reshape_50/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
k

Reshape_50Reshape	split_4:2Reshape_50/shape*
T0*
Tshape0* 
_output_shapes
:

[
Reshape_51/shapeConst*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_51Reshape	split_4:3Reshape_51/shape*
T0*
Tshape0*
_output_shapes	
:
a
Reshape_52/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
j

Reshape_52Reshape	split_4:4Reshape_52/shape*
T0*
Tshape0*
_output_shapes
:	
Z
Reshape_53/shapeConst*
dtype0*
_output_shapes
:*
valueB:
e

Reshape_53Reshape	split_4:5Reshape_53/shape*
_output_shapes
:*
Tshape0*
T0
a
Reshape_54/shapeConst*
dtype0*
_output_shapes
:*
valueB"<      
j

Reshape_54Reshape	split_4:6Reshape_54/shape*
_output_shapes
:	<*
Tshape0*
T0
[
Reshape_55/shapeConst*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_55Reshape	split_4:7Reshape_55/shape*
T0*
_output_shapes	
:*
Tshape0
a
Reshape_56/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_56Reshape	split_4:8Reshape_56/shape*
T0* 
_output_shapes
:
*
Tshape0
[
Reshape_57/shapeConst*
dtype0*
_output_shapes
:*
valueB:
f

Reshape_57Reshape	split_4:9Reshape_57/shape*
_output_shapes	
:*
T0*
Tshape0
a
Reshape_58/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_58Reshape
split_4:10Reshape_58/shape*
T0*
_output_shapes
:	*
Tshape0
Z
Reshape_59/shapeConst*
dtype0*
_output_shapes
:*
valueB:
f

Reshape_59Reshape
split_4:11Reshape_59/shape*
Tshape0*
_output_shapes
:*
T0

beta1_power_2/initial_valueConst*
dtype0* 
_class
loc:@vc/dense/bias*
valueB
 *fff?*
_output_shapes
: 

beta1_power_2
VariableV2*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
shared_name *
	container *
shape: *
dtype0
ś
beta1_power_2/AssignAssignbeta1_power_2beta1_power_2/initial_value*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
use_locking(*
T0
p
beta1_power_2/readIdentitybeta1_power_2* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: 

beta2_power_2/initial_valueConst* 
_class
loc:@vc/dense/bias*
valueB
 *wž?*
_output_shapes
: *
dtype0

beta2_power_2
VariableV2*
_output_shapes
: *
	container * 
_class
loc:@vc/dense/bias*
dtype0*
shape: *
shared_name 
ś
beta2_power_2/AssignAssignbeta2_power_2beta2_power_2/initial_value*
T0*
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
use_locking(
p
beta2_power_2/readIdentitybeta2_power_2* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0
Ť
6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"<      *
dtype0*
_output_shapes
:*"
_class
loc:@vf/dense/kernel

,vf/dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@vf/dense/kernel*
dtype0*
_output_shapes
: 
ô
&vf/dense/kernel/Adam/Initializer/zerosFill6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vf/dense/kernel/Adam/Initializer/zeros/Const*

index_type0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
T0
Ž
vf/dense/kernel/Adam
VariableV2*
_output_shapes
:	<*
dtype0*
shared_name *
	container *"
_class
loc:@vf/dense/kernel*
shape:	<
Ú
vf/dense/kernel/Adam/AssignAssignvf/dense/kernel/Adam&vf/dense/kernel/Adam/Initializer/zeros*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
use_locking(*
T0*
validate_shape(

vf/dense/kernel/Adam/readIdentityvf/dense/kernel/Adam*
_output_shapes
:	<*
T0*"
_class
loc:@vf/dense/kernel
­
8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@vf/dense/kernel*
valueB"<      *
_output_shapes
:*
dtype0

.vf/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0*"
_class
loc:@vf/dense/kernel
ú
(vf/dense/kernel/Adam_1/Initializer/zerosFill8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vf/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<
°
vf/dense/kernel/Adam_1
VariableV2*
	container *
dtype0*
shape:	<*
shared_name *"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<
ŕ
vf/dense/kernel/Adam_1/AssignAssignvf/dense/kernel/Adam_1(vf/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
_output_shapes
:	<*
T0*"
_class
loc:@vf/dense/kernel

vf/dense/kernel/Adam_1/readIdentityvf/dense/kernel/Adam_1*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
T0

$vf/dense/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    * 
_class
loc:@vf/dense/bias*
_output_shapes	
:
˘
vf/dense/bias/Adam
VariableV2* 
_class
loc:@vf/dense/bias*
	container *
dtype0*
_output_shapes	
:*
shape:*
shared_name 
Î
vf/dense/bias/Adam/AssignAssignvf/dense/bias/Adam$vf/dense/bias/Adam/Initializer/zeros* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0

vf/dense/bias/Adam/readIdentityvf/dense/bias/Adam* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:

&vf/dense/bias/Adam_1/Initializer/zerosConst*
valueB*    * 
_class
loc:@vf/dense/bias*
dtype0*
_output_shapes	
:
¤
vf/dense/bias/Adam_1
VariableV2*
shape:*
shared_name * 
_class
loc:@vf/dense/bias*
dtype0*
_output_shapes	
:*
	container 
Ô
vf/dense/bias/Adam_1/AssignAssignvf/dense/bias/Adam_1&vf/dense/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
validate_shape(

vf/dense/bias/Adam_1/readIdentityvf/dense/bias/Adam_1*
T0* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
Ż
8vf/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@vf/dense_1/kernel*
dtype0*
valueB"      *
_output_shapes
:

.vf/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    *$
_class
loc:@vf/dense_1/kernel
ý
(vf/dense_1/kernel/Adam/Initializer/zerosFill8vf/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vf/dense_1/kernel/Adam/Initializer/zeros/Const*$
_class
loc:@vf/dense_1/kernel*

index_type0* 
_output_shapes
:
*
T0
´
vf/dense_1/kernel/Adam
VariableV2*
	container * 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
shape:
*
shared_name *
dtype0
ă
vf/dense_1/kernel/Adam/AssignAssignvf/dense_1/kernel/Adam(vf/dense_1/kernel/Adam/Initializer/zeros* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(

vf/dense_1/kernel/Adam/readIdentityvf/dense_1/kernel/Adam*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:

ą
:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *$
_class
loc:@vf/dense_1/kernel*
dtype0*
_output_shapes
:

0vf/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*$
_class
loc:@vf/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

*vf/dense_1/kernel/Adam_1/Initializer/zerosFill:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vf/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*

index_type0
ś
vf/dense_1/kernel/Adam_1
VariableV2*$
_class
loc:@vf/dense_1/kernel*
dtype0*
shape:
* 
_output_shapes
:
*
	container *
shared_name 
é
vf/dense_1/kernel/Adam_1/AssignAssignvf/dense_1/kernel/Adam_1*vf/dense_1/kernel/Adam_1/Initializer/zeros*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(*
validate_shape(

vf/dense_1/kernel/Adam_1/readIdentityvf/dense_1/kernel/Adam_1* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0

&vf/dense_1/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *"
_class
loc:@vf/dense_1/bias*
dtype0
Ś
vf/dense_1/bias/Adam
VariableV2*
	container *"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
dtype0*
shared_name *
shape:
Ö
vf/dense_1/bias/Adam/AssignAssignvf/dense_1/bias/Adam&vf/dense_1/bias/Adam/Initializer/zeros*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0

vf/dense_1/bias/Adam/readIdentityvf/dense_1/bias/Adam*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias

(vf/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
valueB*    
¨
vf/dense_1/bias/Adam_1
VariableV2*"
_class
loc:@vf/dense_1/bias*
shape:*
	container *
_output_shapes	
:*
shared_name *
dtype0
Ü
vf/dense_1/bias/Adam_1/AssignAssignvf/dense_1/bias/Adam_1(vf/dense_1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0

vf/dense_1/bias/Adam_1/readIdentityvf/dense_1/bias/Adam_1*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0
Ľ
(vf/dense_2/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	*
valueB	*    *$
_class
loc:@vf/dense_2/kernel
˛
vf/dense_2/kernel/Adam
VariableV2*
	container *
dtype0*
shape:	*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
shared_name 
â
vf/dense_2/kernel/Adam/AssignAssignvf/dense_2/kernel/Adam(vf/dense_2/kernel/Adam/Initializer/zeros*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(

vf/dense_2/kernel/Adam/readIdentityvf/dense_2/kernel/Adam*
_output_shapes
:	*
T0*$
_class
loc:@vf/dense_2/kernel
§
*vf/dense_2/kernel/Adam_1/Initializer/zerosConst*
dtype0*
valueB	*    *$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
´
vf/dense_2/kernel/Adam_1
VariableV2*
shared_name *
	container *
dtype0*
shape:	*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
č
vf/dense_2/kernel/Adam_1/AssignAssignvf/dense_2/kernel/Adam_1*vf/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@vf/dense_2/kernel

vf/dense_2/kernel/Adam_1/readIdentityvf/dense_2/kernel/Adam_1*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0

&vf/dense_2/bias/Adam/Initializer/zerosConst*
_output_shapes
:*
dtype0*"
_class
loc:@vf/dense_2/bias*
valueB*    
¤
vf/dense_2/bias/Adam
VariableV2*
shape:*
	container *
dtype0*
shared_name *"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
Ő
vf/dense_2/bias/Adam/AssignAssignvf/dense_2/bias/Adam&vf/dense_2/bias/Adam/Initializer/zeros*
_output_shapes
:*
validate_shape(*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(

vf/dense_2/bias/Adam/readIdentityvf/dense_2/bias/Adam*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:

(vf/dense_2/bias/Adam_1/Initializer/zerosConst*
dtype0*"
_class
loc:@vf/dense_2/bias*
valueB*    *
_output_shapes
:
Ś
vf/dense_2/bias/Adam_1
VariableV2*
dtype0*"
_class
loc:@vf/dense_2/bias*
shared_name *
shape:*
	container *
_output_shapes
:
Ű
vf/dense_2/bias/Adam_1/AssignAssignvf/dense_2/bias/Adam_1(vf/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias

vf/dense_2/bias/Adam_1/readIdentityvf/dense_2/bias/Adam_1*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
Ť
6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*"
_class
loc:@vc/dense/kernel*
valueB"<      

,vc/dense/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *"
_class
loc:@vc/dense/kernel*
valueB
 *    
ô
&vc/dense/kernel/Adam/Initializer/zerosFill6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vc/dense/kernel/Adam/Initializer/zeros/Const*"
_class
loc:@vc/dense/kernel*
T0*

index_type0*
_output_shapes
:	<
Ž
vc/dense/kernel/Adam
VariableV2*
shared_name *
dtype0*
_output_shapes
:	<*
	container *
shape:	<*"
_class
loc:@vc/dense/kernel
Ú
vc/dense/kernel/Adam/AssignAssignvc/dense/kernel/Adam&vc/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
validate_shape(

vc/dense/kernel/Adam/readIdentityvc/dense/kernel/Adam*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<
­
8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*"
_class
loc:@vc/dense/kernel*
valueB"<      

.vc/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    *"
_class
loc:@vc/dense/kernel
ú
(vc/dense/kernel/Adam_1/Initializer/zerosFill8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vc/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*

index_type0
°
vc/dense/kernel/Adam_1
VariableV2*
	container *
shape:	<*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
dtype0*
shared_name 
ŕ
vc/dense/kernel/Adam_1/AssignAssignvc/dense/kernel/Adam_1(vc/dense/kernel/Adam_1/Initializer/zeros*
T0*
validate_shape(*"
_class
loc:@vc/dense/kernel*
use_locking(*
_output_shapes
:	<

vc/dense/kernel/Adam_1/readIdentityvc/dense/kernel/Adam_1*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<

$vc/dense/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*
valueB*    * 
_class
loc:@vc/dense/bias*
dtype0
˘
vc/dense/bias/Adam
VariableV2*
	container *
dtype0*
_output_shapes	
:*
shared_name * 
_class
loc:@vc/dense/bias*
shape:
Î
vc/dense/bias/Adam/AssignAssignvc/dense/bias/Adam$vc/dense/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:

vc/dense/bias/Adam/readIdentityvc/dense/bias/Adam*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias

&vc/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *
dtype0* 
_class
loc:@vc/dense/bias
¤
vc/dense/bias/Adam_1
VariableV2*
shape:*
_output_shapes	
:*
shared_name *
dtype0* 
_class
loc:@vc/dense/bias*
	container 
Ô
vc/dense/bias/Adam_1/AssignAssignvc/dense/bias/Adam_1&vc/dense/bias/Adam_1/Initializer/zeros*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
_output_shapes	
:

vc/dense/bias/Adam_1/readIdentityvc/dense/bias/Adam_1* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:
Ż
8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@vc/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:

.vc/dense_1/kernel/Adam/Initializer/zeros/ConstConst*$
_class
loc:@vc/dense_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
ý
(vc/dense_1/kernel/Adam/Initializer/zerosFill8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vc/dense_1/kernel/Adam/Initializer/zeros/Const*$
_class
loc:@vc/dense_1/kernel*

index_type0* 
_output_shapes
:
*
T0
´
vc/dense_1/kernel/Adam
VariableV2* 
_output_shapes
:
*
shape:
*
	container *
shared_name *$
_class
loc:@vc/dense_1/kernel*
dtype0
ă
vc/dense_1/kernel/Adam/AssignAssignvc/dense_1/kernel/Adam(vc/dense_1/kernel/Adam/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel

vc/dense_1/kernel/Adam/readIdentityvc/dense_1/kernel/Adam* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0
ą
:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *
dtype0*
_output_shapes
:*$
_class
loc:@vc/dense_1/kernel

0vc/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
dtype0*$
_class
loc:@vc/dense_1/kernel*
valueB
 *    

*vc/dense_1/kernel/Adam_1/Initializer/zerosFill:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vc/dense_1/kernel/Adam_1/Initializer/zeros/Const*

index_type0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0
ś
vc/dense_1/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
shared_name *
	container *
shape:

é
vc/dense_1/kernel/Adam_1/AssignAssignvc/dense_1/kernel/Adam_1*vc/dense_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*
T0* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vc/dense_1/kernel

vc/dense_1/kernel/Adam_1/readIdentityvc/dense_1/kernel/Adam_1*
T0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel

&vc/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
valueB*    *
dtype0
Ś
vc/dense_1/bias/Adam
VariableV2*"
_class
loc:@vc/dense_1/bias*
shared_name *
_output_shapes	
:*
shape:*
dtype0*
	container 
Ö
vc/dense_1/bias/Adam/AssignAssignvc/dense_1/bias/Adam&vc/dense_1/bias/Adam/Initializer/zeros*
use_locking(*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:*
T0

vc/dense_1/bias/Adam/readIdentityvc/dense_1/bias/Adam*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:

(vc/dense_1/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@vc/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
¨
vc/dense_1/bias/Adam_1
VariableV2*
	container *
shape:*
shared_name *
dtype0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
Ü
vc/dense_1/bias/Adam_1/AssignAssignvc/dense_1/bias/Adam_1(vc/dense_1/bias/Adam_1/Initializer/zeros*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(

vc/dense_1/bias/Adam_1/readIdentityvc/dense_1/bias/Adam_1*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0
Ľ
(vc/dense_2/kernel/Adam/Initializer/zerosConst*
valueB	*    *
dtype0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
˛
vc/dense_2/kernel/Adam
VariableV2*
dtype0*
	container *$
_class
loc:@vc/dense_2/kernel*
shared_name *
shape:	*
_output_shapes
:	
â
vc/dense_2/kernel/Adam/AssignAssignvc/dense_2/kernel/Adam(vc/dense_2/kernel/Adam/Initializer/zeros*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel

vc/dense_2/kernel/Adam/readIdentityvc/dense_2/kernel/Adam*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel
§
*vc/dense_2/kernel/Adam_1/Initializer/zerosConst*
dtype0*$
_class
loc:@vc/dense_2/kernel*
valueB	*    *
_output_shapes
:	
´
vc/dense_2/kernel/Adam_1
VariableV2*
shape:	*
	container *
_output_shapes
:	*
shared_name *$
_class
loc:@vc/dense_2/kernel*
dtype0
č
vc/dense_2/kernel/Adam_1/AssignAssignvc/dense_2/kernel/Adam_1*vc/dense_2/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
T0

vc/dense_2/kernel/Adam_1/readIdentityvc/dense_2/kernel/Adam_1*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
T0

&vc/dense_2/bias/Adam/Initializer/zerosConst*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
valueB*    *
dtype0
¤
vc/dense_2/bias/Adam
VariableV2*
dtype0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
shape:*
	container *
shared_name 
Ő
vc/dense_2/bias/Adam/AssignAssignvc/dense_2/bias/Adam&vc/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(

vc/dense_2/bias/Adam/readIdentityvc/dense_2/bias/Adam*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias

(vc/dense_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*
dtype0*"
_class
loc:@vc/dense_2/bias*
valueB*    
Ś
vc/dense_2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
	container *
shape:*"
_class
loc:@vc/dense_2/bias
Ű
vc/dense_2/bias/Adam_1/AssignAssignvc/dense_2/bias/Adam_1(vc/dense_2/bias/Adam_1/Initializer/zeros*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(

vc/dense_2/bias/Adam_1/readIdentityvc/dense_2/bias/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias
Y
Adam_2/learning_rateConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Q
Adam_2/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
Q
Adam_2/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *wž?
S
Adam_2/epsilonConst*
_output_shapes
: *
valueB
 *wĚ+2*
dtype0
Ţ
'Adam_2/update_vf/dense/kernel/ApplyAdam	ApplyAdamvf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_48*
use_locking( *
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
T0*
use_nesterov( 
Đ
%Adam_2/update_vf/dense/bias/ApplyAdam	ApplyAdamvf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_49*
use_nesterov( *
T0*
use_locking( *
_output_shapes	
:* 
_class
loc:@vf/dense/bias
é
)Adam_2/update_vf/dense_1/kernel/ApplyAdam	ApplyAdamvf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_50*
T0*
use_locking( *$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
use_nesterov( 
Ú
'Adam_2/update_vf/dense_1/bias/ApplyAdam	ApplyAdamvf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_51*
_output_shapes	
:*
use_locking( *"
_class
loc:@vf/dense_1/bias*
T0*
use_nesterov( 
č
)Adam_2/update_vf/dense_2/kernel/ApplyAdam	ApplyAdamvf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_52*
use_nesterov( *$
_class
loc:@vf/dense_2/kernel*
use_locking( *
_output_shapes
:	*
T0
Ů
'Adam_2/update_vf/dense_2/bias/ApplyAdam	ApplyAdamvf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_53*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
use_locking( *
use_nesterov( *
T0
Ţ
'Adam_2/update_vc/dense/kernel/ApplyAdam	ApplyAdamvc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_54*"
_class
loc:@vc/dense/kernel*
use_nesterov( *
T0*
_output_shapes
:	<*
use_locking( 
Đ
%Adam_2/update_vc/dense/bias/ApplyAdam	ApplyAdamvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_55*
use_locking( *
use_nesterov( *
_output_shapes	
:*
T0* 
_class
loc:@vc/dense/bias
é
)Adam_2/update_vc/dense_1/kernel/ApplyAdam	ApplyAdamvc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_56*$
_class
loc:@vc/dense_1/kernel*
T0*
use_nesterov( * 
_output_shapes
:
*
use_locking( 
Ú
'Adam_2/update_vc/dense_1/bias/ApplyAdam	ApplyAdamvc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_57*"
_class
loc:@vc/dense_1/bias*
use_nesterov( *
T0*
_output_shapes	
:*
use_locking( 
č
)Adam_2/update_vc/dense_2/kernel/ApplyAdam	ApplyAdamvc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_58*
_output_shapes
:	*
use_nesterov( *$
_class
loc:@vc/dense_2/kernel*
T0*
use_locking( 
Ů
'Adam_2/update_vc/dense_2/bias/ApplyAdam	ApplyAdamvc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_59*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:*
use_locking( *
use_nesterov( 
ň

Adam_2/mulMulbeta1_power_2/readAdam_2/beta1&^Adam_2/update_vc/dense/bias/ApplyAdam(^Adam_2/update_vc/dense/kernel/ApplyAdam(^Adam_2/update_vc/dense_1/bias/ApplyAdam*^Adam_2/update_vc/dense_1/kernel/ApplyAdam(^Adam_2/update_vc/dense_2/bias/ApplyAdam*^Adam_2/update_vc/dense_2/kernel/ApplyAdam&^Adam_2/update_vf/dense/bias/ApplyAdam(^Adam_2/update_vf/dense/kernel/ApplyAdam(^Adam_2/update_vf/dense_1/bias/ApplyAdam*^Adam_2/update_vf/dense_1/kernel/ApplyAdam(^Adam_2/update_vf/dense_2/bias/ApplyAdam*^Adam_2/update_vf/dense_2/kernel/ApplyAdam* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: 

Adam_2/AssignAssignbeta1_power_2
Adam_2/mul*
_output_shapes
: *
validate_shape(*
use_locking( *
T0* 
_class
loc:@vc/dense/bias
ô
Adam_2/mul_1Mulbeta2_power_2/readAdam_2/beta2&^Adam_2/update_vc/dense/bias/ApplyAdam(^Adam_2/update_vc/dense/kernel/ApplyAdam(^Adam_2/update_vc/dense_1/bias/ApplyAdam*^Adam_2/update_vc/dense_1/kernel/ApplyAdam(^Adam_2/update_vc/dense_2/bias/ApplyAdam*^Adam_2/update_vc/dense_2/kernel/ApplyAdam&^Adam_2/update_vf/dense/bias/ApplyAdam(^Adam_2/update_vf/dense/kernel/ApplyAdam(^Adam_2/update_vf/dense_1/bias/ApplyAdam*^Adam_2/update_vf/dense_1/kernel/ApplyAdam(^Adam_2/update_vf/dense_2/bias/ApplyAdam*^Adam_2/update_vf/dense_2/kernel/ApplyAdam*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0
˘
Adam_2/Assign_1Assignbeta2_power_2Adam_2/mul_1*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias*
use_locking( *
validate_shape(
Ź
Adam_2NoOp^Adam_2/Assign^Adam_2/Assign_1&^Adam_2/update_vc/dense/bias/ApplyAdam(^Adam_2/update_vc/dense/kernel/ApplyAdam(^Adam_2/update_vc/dense_1/bias/ApplyAdam*^Adam_2/update_vc/dense_1/kernel/ApplyAdam(^Adam_2/update_vc/dense_2/bias/ApplyAdam*^Adam_2/update_vc/dense_2/kernel/ApplyAdam&^Adam_2/update_vf/dense/bias/ApplyAdam(^Adam_2/update_vf/dense/kernel/ApplyAdam(^Adam_2/update_vf/dense_1/bias/ApplyAdam*^Adam_2/update_vf/dense_1/kernel/ApplyAdam(^Adam_2/update_vf/dense_2/bias/ApplyAdam*^Adam_2/update_vf/dense_2/kernel/ApplyAdam
l
Reshape_60/shapeConst^Adam_2*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
q

Reshape_60Reshapevf/dense/kernel/readReshape_60/shape*
_output_shapes	
:x*
Tshape0*
T0
l
Reshape_61/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
o

Reshape_61Reshapevf/dense/bias/readReshape_61/shape*
Tshape0*
T0*
_output_shapes	
:
l
Reshape_62/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
t

Reshape_62Reshapevf/dense_1/kernel/readReshape_62/shape*
T0*
_output_shapes

:*
Tshape0
l
Reshape_63/shapeConst^Adam_2*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
q

Reshape_63Reshapevf/dense_1/bias/readReshape_63/shape*
T0*
Tshape0*
_output_shapes	
:
l
Reshape_64/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s

Reshape_64Reshapevf/dense_2/kernel/readReshape_64/shape*
T0*
_output_shapes	
:*
Tshape0
l
Reshape_65/shapeConst^Adam_2*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
p

Reshape_65Reshapevf/dense_2/bias/readReshape_65/shape*
T0*
Tshape0*
_output_shapes
:
l
Reshape_66/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
q

Reshape_66Reshapevc/dense/kernel/readReshape_66/shape*
_output_shapes	
:x*
Tshape0*
T0
l
Reshape_67/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
o

Reshape_67Reshapevc/dense/bias/readReshape_67/shape*
_output_shapes	
:*
T0*
Tshape0
l
Reshape_68/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
t

Reshape_68Reshapevc/dense_1/kernel/readReshape_68/shape*
T0*
_output_shapes

:*
Tshape0
l
Reshape_69/shapeConst^Adam_2*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
q

Reshape_69Reshapevc/dense_1/bias/readReshape_69/shape*
Tshape0*
_output_shapes	
:*
T0
l
Reshape_70/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
s

Reshape_70Reshapevc/dense_2/kernel/readReshape_70/shape*
_output_shapes	
:*
T0*
Tshape0
l
Reshape_71/shapeConst^Adam_2*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p

Reshape_71Reshapevc/dense_2/bias/readReshape_71/shape*
T0*
_output_shapes
:*
Tshape0
X
concat_5/axisConst^Adam_2*
dtype0*
value	B : *
_output_shapes
: 
ď
concat_5ConcatV2
Reshape_60
Reshape_61
Reshape_62
Reshape_63
Reshape_64
Reshape_65
Reshape_66
Reshape_67
Reshape_68
Reshape_69
Reshape_70
Reshape_71concat_5/axis*
_output_shapes

:ü	*
N*

Tidx0*
T0
h
PyFunc_5PyFuncconcat_5*
Tin
2*
Tout
2*
_output_shapes
:*
token
pyfunc_5

Const_11Const^Adam_2*
dtype0*
_output_shapes
:*E
value<B:"0 <                  <                 
\
split_5/split_dimConst^Adam_2*
dtype0*
_output_shapes
: *
value	B : 
¤
split_5SplitVPyFunc_5Const_11split_5/split_dim*
T0*D
_output_shapes2
0::::::::::::*
	num_split*

Tlen0
j
Reshape_72/shapeConst^Adam_2*
_output_shapes
:*
valueB"<      *
dtype0
h

Reshape_72Reshapesplit_5Reshape_72/shape*
_output_shapes
:	<*
T0*
Tshape0
d
Reshape_73/shapeConst^Adam_2*
_output_shapes
:*
dtype0*
valueB:
f

Reshape_73Reshape	split_5:1Reshape_73/shape*
T0*
_output_shapes	
:*
Tshape0
j
Reshape_74/shapeConst^Adam_2*
dtype0*
valueB"      *
_output_shapes
:
k

Reshape_74Reshape	split_5:2Reshape_74/shape*
Tshape0*
T0* 
_output_shapes
:

d
Reshape_75/shapeConst^Adam_2*
dtype0*
valueB:*
_output_shapes
:
f

Reshape_75Reshape	split_5:3Reshape_75/shape*
Tshape0*
T0*
_output_shapes	
:
j
Reshape_76/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB"      
j

Reshape_76Reshape	split_5:4Reshape_76/shape*
T0*
_output_shapes
:	*
Tshape0
c
Reshape_77/shapeConst^Adam_2*
dtype0*
valueB:*
_output_shapes
:
e

Reshape_77Reshape	split_5:5Reshape_77/shape*
_output_shapes
:*
Tshape0*
T0
j
Reshape_78/shapeConst^Adam_2*
valueB"<      *
dtype0*
_output_shapes
:
j

Reshape_78Reshape	split_5:6Reshape_78/shape*
Tshape0*
T0*
_output_shapes
:	<
d
Reshape_79/shapeConst^Adam_2*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_79Reshape	split_5:7Reshape_79/shape*
T0*
Tshape0*
_output_shapes	
:
j
Reshape_80/shapeConst^Adam_2*
_output_shapes
:*
valueB"      *
dtype0
k

Reshape_80Reshape	split_5:8Reshape_80/shape* 
_output_shapes
:
*
T0*
Tshape0
d
Reshape_81/shapeConst^Adam_2*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_81Reshape	split_5:9Reshape_81/shape*
Tshape0*
_output_shapes	
:*
T0
j
Reshape_82/shapeConst^Adam_2*
_output_shapes
:*
dtype0*
valueB"      
k

Reshape_82Reshape
split_5:10Reshape_82/shape*
_output_shapes
:	*
Tshape0*
T0
c
Reshape_83/shapeConst^Adam_2*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_83Reshape
split_5:11Reshape_83/shape*
T0*
_output_shapes
:*
Tshape0
Ś
Assign_9Assignvf/dense/kernel
Reshape_72*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	<

	Assign_10Assignvf/dense/bias
Reshape_73* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
Ź
	Assign_11Assignvf/dense_1/kernel
Reshape_74*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:

Ł
	Assign_12Assignvf/dense_1/bias
Reshape_75*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:
Ť
	Assign_13Assignvf/dense_2/kernel
Reshape_76*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(*
T0
˘
	Assign_14Assignvf/dense_2/bias
Reshape_77*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias
§
	Assign_15Assignvc/dense/kernel
Reshape_78*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	<

	Assign_16Assignvc/dense/bias
Reshape_79*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
_output_shapes	
:
Ź
	Assign_17Assignvc/dense_1/kernel
Reshape_80*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:
*
validate_shape(
Ł
	Assign_18Assignvc/dense_1/bias
Reshape_81*"
_class
loc:@vc/dense_1/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
Ť
	Assign_19Assignvc/dense_2/kernel
Reshape_82*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
˘
	Assign_20Assignvc/dense_2/bias
Reshape_83*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(
Ź
group_deps_4NoOp^Adam_2
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19
^Assign_20	^Assign_9
,
group_deps_5NoOp^Adam_2^group_deps_4
ż
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta1_power_2/Assign^beta2_power/Assign^beta2_power_1/Assign^beta2_power_2/Assign"^penalty/penalty_param/Adam/Assign$^penalty/penalty_param/Adam_1/Assign$^penalty/penalty_param/Adam_2/Assign$^penalty/penalty_param/Adam_3/Assign^penalty/penalty_param/Assign^pi/dense/bias/Adam/Assign^pi/dense/bias/Adam_1/Assign^pi/dense/bias/Assign^pi/dense/kernel/Adam/Assign^pi/dense/kernel/Adam_1/Assign^pi/dense/kernel/Assign^pi/dense_1/bias/Adam/Assign^pi/dense_1/bias/Adam_1/Assign^pi/dense_1/bias/Assign^pi/dense_1/kernel/Adam/Assign ^pi/dense_1/kernel/Adam_1/Assign^pi/dense_1/kernel/Assign^pi/dense_2/bias/Adam/Assign^pi/dense_2/bias/Adam_1/Assign^pi/dense_2/bias/Assign^pi/dense_2/kernel/Adam/Assign ^pi/dense_2/kernel/Adam_1/Assign^pi/dense_2/kernel/Assign^pi/log_std/Adam/Assign^pi/log_std/Adam_1/Assign^pi/log_std/Assign^vc/dense/bias/Adam/Assign^vc/dense/bias/Adam_1/Assign^vc/dense/bias/Assign^vc/dense/kernel/Adam/Assign^vc/dense/kernel/Adam_1/Assign^vc/dense/kernel/Assign^vc/dense_1/bias/Adam/Assign^vc/dense_1/bias/Adam_1/Assign^vc/dense_1/bias/Assign^vc/dense_1/kernel/Adam/Assign ^vc/dense_1/kernel/Adam_1/Assign^vc/dense_1/kernel/Assign^vc/dense_2/bias/Adam/Assign^vc/dense_2/bias/Adam_1/Assign^vc/dense_2/bias/Assign^vc/dense_2/kernel/Adam/Assign ^vc/dense_2/kernel/Adam_1/Assign^vc/dense_2/kernel/Assign^vf/dense/bias/Adam/Assign^vf/dense/bias/Adam_1/Assign^vf/dense/bias/Assign^vf/dense/kernel/Adam/Assign^vf/dense/kernel/Adam_1/Assign^vf/dense/kernel/Assign^vf/dense_1/bias/Adam/Assign^vf/dense_1/bias/Adam_1/Assign^vf/dense_1/bias/Assign^vf/dense_1/kernel/Adam/Assign ^vf/dense_1/kernel/Adam_1/Assign^vf/dense_1/kernel/Assign^vf/dense_2/bias/Adam/Assign^vf/dense_2/bias/Adam_1/Assign^vf/dense_2/bias/Assign^vf/dense_2/kernel/Adam/Assign ^vf/dense_2/kernel/Adam_1/Assign^vf/dense_2/kernel/Assign
c
Reshape_84/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
q

Reshape_84Reshapepi/dense/kernel/readReshape_84/shape*
Tshape0*
T0*
_output_shapes	
:x
c
Reshape_85/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
o

Reshape_85Reshapepi/dense/bias/readReshape_85/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_86/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
t

Reshape_86Reshapepi/dense_1/kernel/readReshape_86/shape*
Tshape0*
_output_shapes

:*
T0
c
Reshape_87/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
q

Reshape_87Reshapepi/dense_1/bias/readReshape_87/shape*
T0*
_output_shapes	
:*
Tshape0
c
Reshape_88/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
s

Reshape_88Reshapepi/dense_2/kernel/readReshape_88/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_89/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
p

Reshape_89Reshapepi/dense_2/bias/readReshape_89/shape*
Tshape0*
T0*
_output_shapes
:
c
Reshape_90/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
k

Reshape_90Reshapepi/log_std/readReshape_90/shape*
Tshape0*
T0*
_output_shapes
:
c
Reshape_91/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
q

Reshape_91Reshapevf/dense/kernel/readReshape_91/shape*
T0*
_output_shapes	
:x*
Tshape0
c
Reshape_92/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
o

Reshape_92Reshapevf/dense/bias/readReshape_92/shape*
T0*
_output_shapes	
:*
Tshape0
c
Reshape_93/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
t

Reshape_93Reshapevf/dense_1/kernel/readReshape_93/shape*
Tshape0*
T0*
_output_shapes

:
c
Reshape_94/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
q

Reshape_94Reshapevf/dense_1/bias/readReshape_94/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_95/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s

Reshape_95Reshapevf/dense_2/kernel/readReshape_95/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_96/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
p

Reshape_96Reshapevf/dense_2/bias/readReshape_96/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_97/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
q

Reshape_97Reshapevc/dense/kernel/readReshape_97/shape*
Tshape0*
_output_shapes	
:x*
T0
c
Reshape_98/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
o

Reshape_98Reshapevc/dense/bias/readReshape_98/shape*
_output_shapes	
:*
T0*
Tshape0
c
Reshape_99/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
t

Reshape_99Reshapevc/dense_1/kernel/readReshape_99/shape*
Tshape0*
_output_shapes

:*
T0
d
Reshape_100/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
s
Reshape_100Reshapevc/dense_1/bias/readReshape_100/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_101/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
u
Reshape_101Reshapevc/dense_2/kernel/readReshape_101/shape*
Tshape0*
_output_shapes	
:*
T0
d
Reshape_102/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
r
Reshape_102Reshapevc/dense_2/bias/readReshape_102/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_103/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x
Reshape_103Reshapepenalty/penalty_param/readReshape_103/shape*
Tshape0*
_output_shapes
:*
T0
d
Reshape_104/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
n
Reshape_104Reshapebeta1_power/readReshape_104/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_105/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
n
Reshape_105Reshapebeta2_power/readReshape_105/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_106/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
}
Reshape_106Reshapepenalty/penalty_param/Adam/readReshape_106/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_107/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:

Reshape_107Reshape!penalty/penalty_param/Adam_1/readReshape_107/shape*
Tshape0*
_output_shapes
:*
T0
d
Reshape_108/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p
Reshape_108Reshapebeta1_power_1/readReshape_108/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_109/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
p
Reshape_109Reshapebeta2_power_1/readReshape_109/shape*
T0*
_output_shapes
:*
Tshape0
d
Reshape_110/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x
Reshape_110Reshapepi/dense/kernel/Adam/readReshape_110/shape*
T0*
_output_shapes	
:x*
Tshape0
d
Reshape_111/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
z
Reshape_111Reshapepi/dense/kernel/Adam_1/readReshape_111/shape*
_output_shapes	
:x*
Tshape0*
T0
d
Reshape_112/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
v
Reshape_112Reshapepi/dense/bias/Adam/readReshape_112/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_113/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
x
Reshape_113Reshapepi/dense/bias/Adam_1/readReshape_113/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_114/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
{
Reshape_114Reshapepi/dense_1/kernel/Adam/readReshape_114/shape*
Tshape0*
T0*
_output_shapes

:
d
Reshape_115/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
}
Reshape_115Reshapepi/dense_1/kernel/Adam_1/readReshape_115/shape*
Tshape0*
T0*
_output_shapes

:
d
Reshape_116/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
x
Reshape_116Reshapepi/dense_1/bias/Adam/readReshape_116/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_117/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
z
Reshape_117Reshapepi/dense_1/bias/Adam_1/readReshape_117/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_118/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
z
Reshape_118Reshapepi/dense_2/kernel/Adam/readReshape_118/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_119/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
|
Reshape_119Reshapepi/dense_2/kernel/Adam_1/readReshape_119/shape*
T0*
_output_shapes	
:*
Tshape0
d
Reshape_120/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
w
Reshape_120Reshapepi/dense_2/bias/Adam/readReshape_120/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_121/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
y
Reshape_121Reshapepi/dense_2/bias/Adam_1/readReshape_121/shape*
T0*
_output_shapes
:*
Tshape0
d
Reshape_122/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
r
Reshape_122Reshapepi/log_std/Adam/readReshape_122/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_123/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
t
Reshape_123Reshapepi/log_std/Adam_1/readReshape_123/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_124/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙

Reshape_124Reshape!penalty/penalty_param/Adam_2/readReshape_124/shape*
T0*
_output_shapes
:*
Tshape0
d
Reshape_125/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙

Reshape_125Reshape!penalty/penalty_param/Adam_3/readReshape_125/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_126/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p
Reshape_126Reshapebeta1_power_2/readReshape_126/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_127/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
p
Reshape_127Reshapebeta2_power_2/readReshape_127/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_128/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
x
Reshape_128Reshapevf/dense/kernel/Adam/readReshape_128/shape*
Tshape0*
_output_shapes	
:x*
T0
d
Reshape_129/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
z
Reshape_129Reshapevf/dense/kernel/Adam_1/readReshape_129/shape*
T0*
_output_shapes	
:x*
Tshape0
d
Reshape_130/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
v
Reshape_130Reshapevf/dense/bias/Adam/readReshape_130/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_131/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
x
Reshape_131Reshapevf/dense/bias/Adam_1/readReshape_131/shape*
_output_shapes	
:*
Tshape0*
T0
d
Reshape_132/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
{
Reshape_132Reshapevf/dense_1/kernel/Adam/readReshape_132/shape*
Tshape0*
_output_shapes

:*
T0
d
Reshape_133/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
}
Reshape_133Reshapevf/dense_1/kernel/Adam_1/readReshape_133/shape*
_output_shapes

:*
T0*
Tshape0
d
Reshape_134/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
x
Reshape_134Reshapevf/dense_1/bias/Adam/readReshape_134/shape*
_output_shapes	
:*
Tshape0*
T0
d
Reshape_135/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
z
Reshape_135Reshapevf/dense_1/bias/Adam_1/readReshape_135/shape*
T0*
_output_shapes	
:*
Tshape0
d
Reshape_136/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
z
Reshape_136Reshapevf/dense_2/kernel/Adam/readReshape_136/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_137/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
|
Reshape_137Reshapevf/dense_2/kernel/Adam_1/readReshape_137/shape*
Tshape0*
_output_shapes	
:*
T0
d
Reshape_138/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
w
Reshape_138Reshapevf/dense_2/bias/Adam/readReshape_138/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_139/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
y
Reshape_139Reshapevf/dense_2/bias/Adam_1/readReshape_139/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_140/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
x
Reshape_140Reshapevc/dense/kernel/Adam/readReshape_140/shape*
Tshape0*
_output_shapes	
:x*
T0
d
Reshape_141/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
z
Reshape_141Reshapevc/dense/kernel/Adam_1/readReshape_141/shape*
T0*
Tshape0*
_output_shapes	
:x
d
Reshape_142/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
v
Reshape_142Reshapevc/dense/bias/Adam/readReshape_142/shape*
_output_shapes	
:*
Tshape0*
T0
d
Reshape_143/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
x
Reshape_143Reshapevc/dense/bias/Adam_1/readReshape_143/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_144/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
{
Reshape_144Reshapevc/dense_1/kernel/Adam/readReshape_144/shape*
Tshape0*
_output_shapes

:*
T0
d
Reshape_145/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
}
Reshape_145Reshapevc/dense_1/kernel/Adam_1/readReshape_145/shape*
Tshape0*
_output_shapes

:*
T0
d
Reshape_146/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
x
Reshape_146Reshapevc/dense_1/bias/Adam/readReshape_146/shape*
Tshape0*
_output_shapes	
:*
T0
d
Reshape_147/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
z
Reshape_147Reshapevc/dense_1/bias/Adam_1/readReshape_147/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_148/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
z
Reshape_148Reshapevc/dense_2/kernel/Adam/readReshape_148/shape*
T0*
_output_shapes	
:*
Tshape0
d
Reshape_149/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
|
Reshape_149Reshapevc/dense_2/kernel/Adam_1/readReshape_149/shape*
_output_shapes	
:*
Tshape0*
T0
d
Reshape_150/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
w
Reshape_150Reshapevc/dense_2/bias/Adam/readReshape_150/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_151/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
y
Reshape_151Reshapevc/dense_2/bias/Adam_1/readReshape_151/shape*
_output_shapes
:*
Tshape0*
T0
O
concat_6/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ă
concat_6ConcatV2
Reshape_84
Reshape_85
Reshape_86
Reshape_87
Reshape_88
Reshape_89
Reshape_90
Reshape_91
Reshape_92
Reshape_93
Reshape_94
Reshape_95
Reshape_96
Reshape_97
Reshape_98
Reshape_99Reshape_100Reshape_101Reshape_102Reshape_103Reshape_104Reshape_105Reshape_106Reshape_107Reshape_108Reshape_109Reshape_110Reshape_111Reshape_112Reshape_113Reshape_114Reshape_115Reshape_116Reshape_117Reshape_118Reshape_119Reshape_120Reshape_121Reshape_122Reshape_123Reshape_124Reshape_125Reshape_126Reshape_127Reshape_128Reshape_129Reshape_130Reshape_131Reshape_132Reshape_133Reshape_134Reshape_135Reshape_136Reshape_137Reshape_138Reshape_139Reshape_140Reshape_141Reshape_142Reshape_143Reshape_144Reshape_145Reshape_146Reshape_147Reshape_148Reshape_149Reshape_150Reshape_151concat_6/axis*

Tidx0*
_output_shapes

:ô,*
ND*
T0
h
PyFunc_6PyFuncconcat_6*
Tout
2*
token
pyfunc_6*
Tin
2*
_output_shapes
:
ĺ
Const_12Const*
dtype0*
_output_shapes
:D*¨
valueBD" <                     <                  <                                       <   <                                                   <   <                                 <   <                                
S
split_6/split_dimConst*
value	B : *
_output_shapes
: *
dtype0

split_6SplitVPyFunc_6Const_12split_6/split_dim*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*

Tlen0*
	num_splitD*
T0
b
Reshape_152/shapeConst*
_output_shapes
:*
dtype0*
valueB"<      
j
Reshape_152Reshapesplit_6Reshape_152/shape*
Tshape0*
_output_shapes
:	<*
T0
\
Reshape_153/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_153Reshape	split_6:1Reshape_153/shape*
T0*
_output_shapes	
:*
Tshape0
b
Reshape_154/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
m
Reshape_154Reshape	split_6:2Reshape_154/shape*
Tshape0* 
_output_shapes
:
*
T0
\
Reshape_155/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_155Reshape	split_6:3Reshape_155/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_156/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
l
Reshape_156Reshape	split_6:4Reshape_156/shape*
Tshape0*
T0*
_output_shapes
:	
[
Reshape_157/shapeConst*
dtype0*
_output_shapes
:*
valueB:
g
Reshape_157Reshape	split_6:5Reshape_157/shape*
Tshape0*
T0*
_output_shapes
:
[
Reshape_158/shapeConst*
dtype0*
_output_shapes
:*
valueB:
g
Reshape_158Reshape	split_6:6Reshape_158/shape*
T0*
_output_shapes
:*
Tshape0
b
Reshape_159/shapeConst*
valueB"<      *
_output_shapes
:*
dtype0
l
Reshape_159Reshape	split_6:7Reshape_159/shape*
T0*
_output_shapes
:	<*
Tshape0
\
Reshape_160/shapeConst*
_output_shapes
:*
dtype0*
valueB:
h
Reshape_160Reshape	split_6:8Reshape_160/shape*
Tshape0*
T0*
_output_shapes	
:
b
Reshape_161/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
m
Reshape_161Reshape	split_6:9Reshape_161/shape* 
_output_shapes
:
*
T0*
Tshape0
\
Reshape_162/shapeConst*
valueB:*
_output_shapes
:*
dtype0
i
Reshape_162Reshape
split_6:10Reshape_162/shape*
Tshape0*
T0*
_output_shapes	
:
b
Reshape_163/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
m
Reshape_163Reshape
split_6:11Reshape_163/shape*
T0*
Tshape0*
_output_shapes
:	
[
Reshape_164/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_164Reshape
split_6:12Reshape_164/shape*
_output_shapes
:*
Tshape0*
T0
b
Reshape_165/shapeConst*
dtype0*
_output_shapes
:*
valueB"<      
m
Reshape_165Reshape
split_6:13Reshape_165/shape*
Tshape0*
_output_shapes
:	<*
T0
\
Reshape_166/shapeConst*
dtype0*
valueB:*
_output_shapes
:
i
Reshape_166Reshape
split_6:14Reshape_166/shape*
Tshape0*
_output_shapes	
:*
T0
b
Reshape_167/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
n
Reshape_167Reshape
split_6:15Reshape_167/shape*
Tshape0* 
_output_shapes
:
*
T0
\
Reshape_168/shapeConst*
_output_shapes
:*
dtype0*
valueB:
i
Reshape_168Reshape
split_6:16Reshape_168/shape*
Tshape0*
_output_shapes	
:*
T0
b
Reshape_169/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_169Reshape
split_6:17Reshape_169/shape*
T0*
_output_shapes
:	*
Tshape0
[
Reshape_170/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_170Reshape
split_6:18Reshape_170/shape*
Tshape0*
T0*
_output_shapes
:
T
Reshape_171/shapeConst*
_output_shapes
: *
valueB *
dtype0
d
Reshape_171Reshape
split_6:19Reshape_171/shape*
T0*
Tshape0*
_output_shapes
: 
T
Reshape_172/shapeConst*
_output_shapes
: *
dtype0*
valueB 
d
Reshape_172Reshape
split_6:20Reshape_172/shape*
Tshape0*
_output_shapes
: *
T0
T
Reshape_173/shapeConst*
valueB *
_output_shapes
: *
dtype0
d
Reshape_173Reshape
split_6:21Reshape_173/shape*
Tshape0*
_output_shapes
: *
T0
T
Reshape_174/shapeConst*
dtype0*
valueB *
_output_shapes
: 
d
Reshape_174Reshape
split_6:22Reshape_174/shape*
T0*
_output_shapes
: *
Tshape0
T
Reshape_175/shapeConst*
dtype0*
valueB *
_output_shapes
: 
d
Reshape_175Reshape
split_6:23Reshape_175/shape*
_output_shapes
: *
T0*
Tshape0
T
Reshape_176/shapeConst*
dtype0*
valueB *
_output_shapes
: 
d
Reshape_176Reshape
split_6:24Reshape_176/shape*
Tshape0*
_output_shapes
: *
T0
T
Reshape_177/shapeConst*
dtype0*
valueB *
_output_shapes
: 
d
Reshape_177Reshape
split_6:25Reshape_177/shape*
Tshape0*
_output_shapes
: *
T0
b
Reshape_178/shapeConst*
dtype0*
valueB"<      *
_output_shapes
:
m
Reshape_178Reshape
split_6:26Reshape_178/shape*
_output_shapes
:	<*
T0*
Tshape0
b
Reshape_179/shapeConst*
valueB"<      *
_output_shapes
:*
dtype0
m
Reshape_179Reshape
split_6:27Reshape_179/shape*
T0*
_output_shapes
:	<*
Tshape0
\
Reshape_180/shapeConst*
_output_shapes
:*
valueB:*
dtype0
i
Reshape_180Reshape
split_6:28Reshape_180/shape*
_output_shapes	
:*
T0*
Tshape0
\
Reshape_181/shapeConst*
_output_shapes
:*
dtype0*
valueB:
i
Reshape_181Reshape
split_6:29Reshape_181/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_182/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
n
Reshape_182Reshape
split_6:30Reshape_182/shape* 
_output_shapes
:
*
Tshape0*
T0
b
Reshape_183/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
n
Reshape_183Reshape
split_6:31Reshape_183/shape*
T0* 
_output_shapes
:
*
Tshape0
\
Reshape_184/shapeConst*
dtype0*
valueB:*
_output_shapes
:
i
Reshape_184Reshape
split_6:32Reshape_184/shape*
_output_shapes	
:*
Tshape0*
T0
\
Reshape_185/shapeConst*
_output_shapes
:*
dtype0*
valueB:
i
Reshape_185Reshape
split_6:33Reshape_185/shape*
Tshape0*
_output_shapes	
:*
T0
b
Reshape_186/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
m
Reshape_186Reshape
split_6:34Reshape_186/shape*
T0*
Tshape0*
_output_shapes
:	
b
Reshape_187/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
m
Reshape_187Reshape
split_6:35Reshape_187/shape*
T0*
Tshape0*
_output_shapes
:	
[
Reshape_188/shapeConst*
valueB:*
_output_shapes
:*
dtype0
h
Reshape_188Reshape
split_6:36Reshape_188/shape*
T0*
_output_shapes
:*
Tshape0
[
Reshape_189/shapeConst*
_output_shapes
:*
dtype0*
valueB:
h
Reshape_189Reshape
split_6:37Reshape_189/shape*
Tshape0*
_output_shapes
:*
T0
[
Reshape_190/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_190Reshape
split_6:38Reshape_190/shape*
_output_shapes
:*
T0*
Tshape0
[
Reshape_191/shapeConst*
valueB:*
_output_shapes
:*
dtype0
h
Reshape_191Reshape
split_6:39Reshape_191/shape*
T0*
Tshape0*
_output_shapes
:
T
Reshape_192/shapeConst*
valueB *
_output_shapes
: *
dtype0
d
Reshape_192Reshape
split_6:40Reshape_192/shape*
_output_shapes
: *
T0*
Tshape0
T
Reshape_193/shapeConst*
_output_shapes
: *
dtype0*
valueB 
d
Reshape_193Reshape
split_6:41Reshape_193/shape*
Tshape0*
T0*
_output_shapes
: 
T
Reshape_194/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_194Reshape
split_6:42Reshape_194/shape*
Tshape0*
T0*
_output_shapes
: 
T
Reshape_195/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_195Reshape
split_6:43Reshape_195/shape*
Tshape0*
_output_shapes
: *
T0
b
Reshape_196/shapeConst*
dtype0*
_output_shapes
:*
valueB"<      
m
Reshape_196Reshape
split_6:44Reshape_196/shape*
T0*
Tshape0*
_output_shapes
:	<
b
Reshape_197/shapeConst*
dtype0*
_output_shapes
:*
valueB"<      
m
Reshape_197Reshape
split_6:45Reshape_197/shape*
_output_shapes
:	<*
T0*
Tshape0
\
Reshape_198/shapeConst*
_output_shapes
:*
dtype0*
valueB:
i
Reshape_198Reshape
split_6:46Reshape_198/shape*
T0*
_output_shapes	
:*
Tshape0
\
Reshape_199/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_199Reshape
split_6:47Reshape_199/shape*
T0*
_output_shapes	
:*
Tshape0
b
Reshape_200/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
n
Reshape_200Reshape
split_6:48Reshape_200/shape*
Tshape0*
T0* 
_output_shapes
:

b
Reshape_201/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
n
Reshape_201Reshape
split_6:49Reshape_201/shape* 
_output_shapes
:
*
Tshape0*
T0
\
Reshape_202/shapeConst*
_output_shapes
:*
valueB:*
dtype0
i
Reshape_202Reshape
split_6:50Reshape_202/shape*
_output_shapes	
:*
Tshape0*
T0
\
Reshape_203/shapeConst*
_output_shapes
:*
dtype0*
valueB:
i
Reshape_203Reshape
split_6:51Reshape_203/shape*
T0*
_output_shapes	
:*
Tshape0
b
Reshape_204/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
m
Reshape_204Reshape
split_6:52Reshape_204/shape*
Tshape0*
_output_shapes
:	*
T0
b
Reshape_205/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
m
Reshape_205Reshape
split_6:53Reshape_205/shape*
_output_shapes
:	*
T0*
Tshape0
[
Reshape_206/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_206Reshape
split_6:54Reshape_206/shape*
_output_shapes
:*
Tshape0*
T0
[
Reshape_207/shapeConst*
_output_shapes
:*
dtype0*
valueB:
h
Reshape_207Reshape
split_6:55Reshape_207/shape*
T0*
_output_shapes
:*
Tshape0
b
Reshape_208/shapeConst*
valueB"<      *
dtype0*
_output_shapes
:
m
Reshape_208Reshape
split_6:56Reshape_208/shape*
T0*
_output_shapes
:	<*
Tshape0
b
Reshape_209/shapeConst*
dtype0*
_output_shapes
:*
valueB"<      
m
Reshape_209Reshape
split_6:57Reshape_209/shape*
_output_shapes
:	<*
Tshape0*
T0
\
Reshape_210/shapeConst*
dtype0*
valueB:*
_output_shapes
:
i
Reshape_210Reshape
split_6:58Reshape_210/shape*
_output_shapes	
:*
Tshape0*
T0
\
Reshape_211/shapeConst*
_output_shapes
:*
dtype0*
valueB:
i
Reshape_211Reshape
split_6:59Reshape_211/shape*
_output_shapes	
:*
Tshape0*
T0
b
Reshape_212/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
n
Reshape_212Reshape
split_6:60Reshape_212/shape*
Tshape0* 
_output_shapes
:
*
T0
b
Reshape_213/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
n
Reshape_213Reshape
split_6:61Reshape_213/shape* 
_output_shapes
:
*
Tshape0*
T0
\
Reshape_214/shapeConst*
dtype0*
_output_shapes
:*
valueB:
i
Reshape_214Reshape
split_6:62Reshape_214/shape*
_output_shapes	
:*
Tshape0*
T0
\
Reshape_215/shapeConst*
valueB:*
_output_shapes
:*
dtype0
i
Reshape_215Reshape
split_6:63Reshape_215/shape*
Tshape0*
_output_shapes	
:*
T0
b
Reshape_216/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
m
Reshape_216Reshape
split_6:64Reshape_216/shape*
_output_shapes
:	*
Tshape0*
T0
b
Reshape_217/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
m
Reshape_217Reshape
split_6:65Reshape_217/shape*
_output_shapes
:	*
Tshape0*
T0
[
Reshape_218/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_218Reshape
split_6:66Reshape_218/shape*
T0*
Tshape0*
_output_shapes
:
[
Reshape_219/shapeConst*
dtype0*
_output_shapes
:*
valueB:
h
Reshape_219Reshape
split_6:67Reshape_219/shape*
T0*
Tshape0*
_output_shapes
:
¨
	Assign_21Assignpi/dense/kernelReshape_152*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	<
 
	Assign_22Assignpi/dense/biasReshape_153*
T0*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:
­
	Assign_23Assignpi/dense_1/kernelReshape_154*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(
¤
	Assign_24Assignpi/dense_1/biasReshape_155*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(
Ź
	Assign_25Assignpi/dense_2/kernelReshape_156*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	
Ł
	Assign_26Assignpi/dense_2/biasReshape_157*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:

	Assign_27Assign
pi/log_stdReshape_158*
validate_shape(*
_output_shapes
:*
T0*
_class
loc:@pi/log_std*
use_locking(
¨
	Assign_28Assignvf/dense/kernelReshape_159*
validate_shape(*
T0*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
use_locking(
 
	Assign_29Assignvf/dense/biasReshape_160*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(*
T0
­
	Assign_30Assignvf/dense_1/kernelReshape_161*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
T0*
validate_shape(
¤
	Assign_31Assignvf/dense_1/biasReshape_162*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(
Ź
	Assign_32Assignvf/dense_2/kernelReshape_163*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0
Ł
	Assign_33Assignvf/dense_2/biasReshape_164*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
¨
	Assign_34Assignvc/dense/kernelReshape_165*"
_class
loc:@vc/dense/kernel*
T0*
use_locking(*
_output_shapes
:	<*
validate_shape(
 
	Assign_35Assignvc/dense/biasReshape_166*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
­
	Assign_36Assignvc/dense_1/kernelReshape_167* 
_output_shapes
:
*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0
¤
	Assign_37Assignvc/dense_1/biasReshape_168*
T0*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:
Ź
	Assign_38Assignvc/dense_2/kernelReshape_169*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
use_locking(*
validate_shape(
Ł
	Assign_39Assignvc/dense_2/biasReshape_170*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
Ť
	Assign_40Assignpenalty/penalty_paramReshape_171*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: *
T0
Ą
	Assign_41Assignbeta1_powerReshape_172*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(*
T0*
validate_shape(
Ą
	Assign_42Assignbeta2_powerReshape_173*(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
°
	Assign_43Assignpenalty/penalty_param/AdamReshape_174*
_output_shapes
: *
use_locking(*
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param
˛
	Assign_44Assignpenalty/penalty_param/Adam_1Reshape_175*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
_output_shapes
: 
Ł
	Assign_45Assignbeta1_power_1Reshape_176*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0
Ł
	Assign_46Assignbeta2_power_1Reshape_177*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
_output_shapes
: *
T0
­
	Assign_47Assignpi/dense/kernel/AdamReshape_178*
_output_shapes
:	<*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(
Ż
	Assign_48Assignpi/dense/kernel/Adam_1Reshape_179*
T0*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(
Ľ
	Assign_49Assignpi/dense/bias/AdamReshape_180* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
§
	Assign_50Assignpi/dense/bias/Adam_1Reshape_181*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
T0
˛
	Assign_51Assignpi/dense_1/kernel/AdamReshape_182*
validate_shape(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0
´
	Assign_52Assignpi/dense_1/kernel/Adam_1Reshape_183*
validate_shape(*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
T0
Š
	Assign_53Assignpi/dense_1/bias/AdamReshape_184*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes	
:
Ť
	Assign_54Assignpi/dense_1/bias/Adam_1Reshape_185*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
ą
	Assign_55Assignpi/dense_2/kernel/AdamReshape_186*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
use_locking(*
validate_shape(
ł
	Assign_56Assignpi/dense_2/kernel/Adam_1Reshape_187*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes
:	*
T0
¨
	Assign_57Assignpi/dense_2/bias/AdamReshape_188*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias
Ş
	Assign_58Assignpi/dense_2/bias/Adam_1Reshape_189*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(

	Assign_59Assignpi/log_std/AdamReshape_190*
_class
loc:@pi/log_std*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
 
	Assign_60Assignpi/log_std/Adam_1Reshape_191*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*
_class
loc:@pi/log_std
˛
	Assign_61Assignpenalty/penalty_param/Adam_2Reshape_192*
validate_shape(*
use_locking(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param
˛
	Assign_62Assignpenalty/penalty_param/Adam_3Reshape_193*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(

	Assign_63Assignbeta1_power_2Reshape_194* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
validate_shape(*
T0*
use_locking(

	Assign_64Assignbeta2_power_2Reshape_195*
validate_shape(*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias*
use_locking(
­
	Assign_65Assignvf/dense/kernel/AdamReshape_196*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	<
Ż
	Assign_66Assignvf/dense/kernel/Adam_1Reshape_197*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
validate_shape(
Ľ
	Assign_67Assignvf/dense/bias/AdamReshape_198*
_output_shapes	
:*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias*
T0
§
	Assign_68Assignvf/dense/bias/Adam_1Reshape_199*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias
˛
	Assign_69Assignvf/dense_1/kernel/AdamReshape_200*
T0* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(
´
	Assign_70Assignvf/dense_1/kernel/Adam_1Reshape_201*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
T0*
use_locking(
Š
	Assign_71Assignvf/dense_1/bias/AdamReshape_202*
use_locking(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0*
validate_shape(
Ť
	Assign_72Assignvf/dense_1/bias/Adam_1Reshape_203*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(
ą
	Assign_73Assignvf/dense_2/kernel/AdamReshape_204*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
T0
ł
	Assign_74Assignvf/dense_2/kernel/Adam_1Reshape_205*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
validate_shape(
¨
	Assign_75Assignvf/dense_2/bias/AdamReshape_206*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
Ş
	Assign_76Assignvf/dense_2/bias/Adam_1Reshape_207*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:
­
	Assign_77Assignvc/dense/kernel/AdamReshape_208*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
T0*
use_locking(
Ż
	Assign_78Assignvc/dense/kernel/Adam_1Reshape_209*
_output_shapes
:	<*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense/kernel*
T0
Ľ
	Assign_79Assignvc/dense/bias/AdamReshape_210*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
use_locking(*
T0
§
	Assign_80Assignvc/dense/bias/Adam_1Reshape_211*
use_locking(*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:
˛
	Assign_81Assignvc/dense_1/kernel/AdamReshape_212* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(*
validate_shape(
´
	Assign_82Assignvc/dense_1/kernel/Adam_1Reshape_213*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
T0
Š
	Assign_83Assignvc/dense_1/bias/AdamReshape_214*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias
Ť
	Assign_84Assignvc/dense_1/bias/Adam_1Reshape_215*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias
ą
	Assign_85Assignvc/dense_2/kernel/AdamReshape_216*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
ł
	Assign_86Assignvc/dense_2/kernel/Adam_1Reshape_217*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0
¨
	Assign_87Assignvc/dense_2/bias/AdamReshape_218*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias
Ş
	Assign_88Assignvc/dense_2/bias/Adam_1Reshape_219*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
use_locking(
Ä
group_deps_6NoOp
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
^Assign_63
^Assign_64
^Assign_65
^Assign_66
^Assign_67
^Assign_68
^Assign_69
^Assign_70
^Assign_71
^Assign_72
^Assign_73
^Assign_74
^Assign_75
^Assign_76
^Assign_77
^Assign_78
^Assign_79
^Assign_80
^Assign_81
^Assign_82
^Assign_83
^Assign_84
^Assign_85
^Assign_86
^Assign_87
^Assign_88
Y
save/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_504519c7f7414656a5b70995228b1bbd/part*
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
\
save/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
ő
save/SaveV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D*
dtype0
î
save/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ę
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
T0*
_output_shapes
:*

axis 
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
ř
save/RestoreV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
ń
save/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D*
dtype0
â
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ś
save/AssignAssignbeta1_powersave/RestoreV2*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(*
use_locking(
Ź
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(*
_output_shapes
: *
T0
¤
save/Assign_2Assignbeta1_power_2save/RestoreV2:2*
T0*
use_locking(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
validate_shape(
Ş
save/Assign_3Assignbeta2_powersave/RestoreV2:3*
_output_shapes
: *
validate_shape(*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param
Ź
save/Assign_4Assignbeta2_power_1save/RestoreV2:4*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0*
use_locking(
¤
save/Assign_5Assignbeta2_power_2save/RestoreV2:5*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@vc/dense/bias
´
save/Assign_6Assignpenalty/penalty_paramsave/RestoreV2:6*
_output_shapes
: *
use_locking(*
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param
š
save/Assign_7Assignpenalty/penalty_param/Adamsave/RestoreV2:7*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(*
use_locking(
ť
save/Assign_8Assignpenalty/penalty_param/Adam_1save/RestoreV2:8*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0
ť
save/Assign_9Assignpenalty/penalty_param/Adam_2save/RestoreV2:9*
use_locking(*
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
˝
save/Assign_10Assignpenalty/penalty_param/Adam_3save/RestoreV2:10*
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(
Ť
save/Assign_11Assignpi/dense/biassave/RestoreV2:11* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:
°
save/Assign_12Assignpi/dense/bias/Adamsave/RestoreV2:12*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
T0
˛
save/Assign_13Assignpi/dense/bias/Adam_1save/RestoreV2:13*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
use_locking(
ł
save/Assign_14Assignpi/dense/kernelsave/RestoreV2:14*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
T0*
validate_shape(*
use_locking(
¸
save/Assign_15Assignpi/dense/kernel/Adamsave/RestoreV2:15*
_output_shapes
:	<*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel
ş
save/Assign_16Assignpi/dense/kernel/Adam_1save/RestoreV2:16*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	<*
validate_shape(*
T0
Ż
save/Assign_17Assignpi/dense_1/biassave/RestoreV2:17*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
use_locking(
´
save/Assign_18Assignpi/dense_1/bias/Adamsave/RestoreV2:18*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
ś
save/Assign_19Assignpi/dense_1/bias/Adam_1save/RestoreV2:19*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
use_locking(
¸
save/Assign_20Assignpi/dense_1/kernelsave/RestoreV2:20* 
_output_shapes
:
*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(
˝
save/Assign_21Assignpi/dense_1/kernel/Adamsave/RestoreV2:21*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

ż
save/Assign_22Assignpi/dense_1/kernel/Adam_1save/RestoreV2:22*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
T0
Ž
save/Assign_23Assignpi/dense_2/biassave/RestoreV2:23*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0
ł
save/Assign_24Assignpi/dense_2/bias/Adamsave/RestoreV2:24*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
ľ
save/Assign_25Assignpi/dense_2/bias/Adam_1save/RestoreV2:25*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
ˇ
save/Assign_26Assignpi/dense_2/kernelsave/RestoreV2:26*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes
:	*
T0
ź
save/Assign_27Assignpi/dense_2/kernel/Adamsave/RestoreV2:27*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
use_locking(
ž
save/Assign_28Assignpi/dense_2/kernel/Adam_1save/RestoreV2:28*
_output_shapes
:	*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(
¤
save/Assign_29Assign
pi/log_stdsave/RestoreV2:29*
_output_shapes
:*
_class
loc:@pi/log_std*
T0*
use_locking(*
validate_shape(
Š
save/Assign_30Assignpi/log_std/Adamsave/RestoreV2:30*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*
_class
loc:@pi/log_std
Ť
save/Assign_31Assignpi/log_std/Adam_1save/RestoreV2:31*
_class
loc:@pi/log_std*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
Ť
save/Assign_32Assignvc/dense/biassave/RestoreV2:32*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:
°
save/Assign_33Assignvc/dense/bias/Adamsave/RestoreV2:33*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0*
use_locking(
˛
save/Assign_34Assignvc/dense/bias/Adam_1save/RestoreV2:34*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0
ł
save/Assign_35Assignvc/dense/kernelsave/RestoreV2:35*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel
¸
save/Assign_36Assignvc/dense/kernel/Adamsave/RestoreV2:36*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
validate_shape(*
use_locking(*
T0
ş
save/Assign_37Assignvc/dense/kernel/Adam_1save/RestoreV2:37*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<*
validate_shape(*
use_locking(
Ż
save/Assign_38Assignvc/dense_1/biassave/RestoreV2:38*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias
´
save/Assign_39Assignvc/dense_1/bias/Adamsave/RestoreV2:39*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0
ś
save/Assign_40Assignvc/dense_1/bias/Adam_1save/RestoreV2:40*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(
¸
save/Assign_41Assignvc/dense_1/kernelsave/RestoreV2:41*
use_locking(*
validate_shape(* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel
˝
save/Assign_42Assignvc/dense_1/kernel/Adamsave/RestoreV2:42*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(
ż
save/Assign_43Assignvc/dense_1/kernel/Adam_1save/RestoreV2:43*
validate_shape(*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0
Ž
save/Assign_44Assignvc/dense_2/biassave/RestoreV2:44*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:
ł
save/Assign_45Assignvc/dense_2/bias/Adamsave/RestoreV2:45*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(
ľ
save/Assign_46Assignvc/dense_2/bias/Adam_1save/RestoreV2:46*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
T0
ˇ
save/Assign_47Assignvc/dense_2/kernelsave/RestoreV2:47*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
T0
ź
save/Assign_48Assignvc/dense_2/kernel/Adamsave/RestoreV2:48*
validate_shape(*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0
ž
save/Assign_49Assignvc/dense_2/kernel/Adam_1save/RestoreV2:49*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0
Ť
save/Assign_50Assignvf/dense/biassave/RestoreV2:50*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0*
validate_shape(
°
save/Assign_51Assignvf/dense/bias/Adamsave/RestoreV2:51*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(
˛
save/Assign_52Assignvf/dense/bias/Adam_1save/RestoreV2:52*
_output_shapes	
:*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias*
T0
ł
save/Assign_53Assignvf/dense/kernelsave/RestoreV2:53*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel
¸
save/Assign_54Assignvf/dense/kernel/Adamsave/RestoreV2:54*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	<*
T0
ş
save/Assign_55Assignvf/dense/kernel/Adam_1save/RestoreV2:55*
validate_shape(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
use_locking(*
T0
Ż
save/Assign_56Assignvf/dense_1/biassave/RestoreV2:56*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
´
save/Assign_57Assignvf/dense_1/bias/Adamsave/RestoreV2:57*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:
ś
save/Assign_58Assignvf/dense_1/bias/Adam_1save/RestoreV2:58*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:
¸
save/Assign_59Assignvf/dense_1/kernelsave/RestoreV2:59*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0
˝
save/Assign_60Assignvf/dense_1/kernel/Adamsave/RestoreV2:60*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(
ż
save/Assign_61Assignvf/dense_1/kernel/Adam_1save/RestoreV2:61*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:

Ž
save/Assign_62Assignvf/dense_2/biassave/RestoreV2:62*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(
ł
save/Assign_63Assignvf/dense_2/bias/Adamsave/RestoreV2:63*
use_locking(*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:
ľ
save/Assign_64Assignvf/dense_2/bias/Adam_1save/RestoreV2:64*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(
ˇ
save/Assign_65Assignvf/dense_2/kernelsave/RestoreV2:65*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	
ź
save/Assign_66Assignvf/dense_2/kernel/Adamsave/RestoreV2:66*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0*
use_locking(*
validate_shape(
ž
save/Assign_67Assignvf/dense_2/kernel/Adam_1save/RestoreV2:67*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
	
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
_output_shapes
: *
dtype0*
shape: 

save_1/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_d183301992be409ebb1afc81adcb3bd4/part*
dtype0
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
S
save_1/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_1/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
÷
save_1/SaveV2/tensor_namesConst*
_output_shapes
:D*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
đ
save_1/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ň
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: *
T0
Ł
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
T0*
N*
_output_shapes
:*

axis 

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
ú
save_1/RestoreV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
ó
!save_1/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ę
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ş
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(*
_output_shapes
: 
°
save_1/Assign_1Assignbeta1_power_1save_1/RestoreV2:1*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
T0*
use_locking(
¨
save_1/Assign_2Assignbeta1_power_2save_1/RestoreV2:2* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
: 
Ž
save_1/Assign_3Assignbeta2_powersave_1/RestoreV2:3*
use_locking(*
_output_shapes
: *
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param
°
save_1/Assign_4Assignbeta2_power_1save_1/RestoreV2:4*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0
¨
save_1/Assign_5Assignbeta2_power_2save_1/RestoreV2:5* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
¸
save_1/Assign_6Assignpenalty/penalty_paramsave_1/RestoreV2:6*
T0*
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
˝
save_1/Assign_7Assignpenalty/penalty_param/Adamsave_1/RestoreV2:7*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0
ż
save_1/Assign_8Assignpenalty/penalty_param/Adam_1save_1/RestoreV2:8*
_output_shapes
: *
T0*
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(
ż
save_1/Assign_9Assignpenalty/penalty_param/Adam_2save_1/RestoreV2:9*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(*
T0*
use_locking(
Á
save_1/Assign_10Assignpenalty/penalty_param/Adam_3save_1/RestoreV2:10*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: *
T0
Ż
save_1/Assign_11Assignpi/dense/biassave_1/RestoreV2:11*
_output_shapes	
:*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
use_locking(
´
save_1/Assign_12Assignpi/dense/bias/Adamsave_1/RestoreV2:12*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(* 
_class
loc:@pi/dense/bias
ś
save_1/Assign_13Assignpi/dense/bias/Adam_1save_1/RestoreV2:13* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
ˇ
save_1/Assign_14Assignpi/dense/kernelsave_1/RestoreV2:14*
validate_shape(*
_output_shapes
:	<*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(
ź
save_1/Assign_15Assignpi/dense/kernel/Adamsave_1/RestoreV2:15*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(*
_output_shapes
:	<
ž
save_1/Assign_16Assignpi/dense/kernel/Adam_1save_1/RestoreV2:16*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel
ł
save_1/Assign_17Assignpi/dense_1/biassave_1/RestoreV2:17*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
¸
save_1/Assign_18Assignpi/dense_1/bias/Adamsave_1/RestoreV2:18*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
use_locking(
ş
save_1/Assign_19Assignpi/dense_1/bias/Adam_1save_1/RestoreV2:19*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:
ź
save_1/Assign_20Assignpi/dense_1/kernelsave_1/RestoreV2:20*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
validate_shape(*
T0
Á
save_1/Assign_21Assignpi/dense_1/kernel/Adamsave_1/RestoreV2:21*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:

Ă
save_1/Assign_22Assignpi/dense_1/kernel/Adam_1save_1/RestoreV2:22*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel
˛
save_1/Assign_23Assignpi/dense_2/biassave_1/RestoreV2:23*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0
ˇ
save_1/Assign_24Assignpi/dense_2/bias/Adamsave_1/RestoreV2:24*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
š
save_1/Assign_25Assignpi/dense_2/bias/Adam_1save_1/RestoreV2:25*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(
ť
save_1/Assign_26Assignpi/dense_2/kernelsave_1/RestoreV2:26*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(
Ŕ
save_1/Assign_27Assignpi/dense_2/kernel/Adamsave_1/RestoreV2:27*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0
Â
save_1/Assign_28Assignpi/dense_2/kernel/Adam_1save_1/RestoreV2:28*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0
¨
save_1/Assign_29Assign
pi/log_stdsave_1/RestoreV2:29*
_class
loc:@pi/log_std*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
­
save_1/Assign_30Assignpi/log_std/Adamsave_1/RestoreV2:30*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*
_class
loc:@pi/log_std
Ż
save_1/Assign_31Assignpi/log_std/Adam_1save_1/RestoreV2:31*
use_locking(*
_class
loc:@pi/log_std*
validate_shape(*
T0*
_output_shapes
:
Ż
save_1/Assign_32Assignvc/dense/biassave_1/RestoreV2:32*
_output_shapes	
:*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(
´
save_1/Assign_33Assignvc/dense/bias/Adamsave_1/RestoreV2:33* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
ś
save_1/Assign_34Assignvc/dense/bias/Adam_1save_1/RestoreV2:34*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(* 
_class
loc:@vc/dense/bias
ˇ
save_1/Assign_35Assignvc/dense/kernelsave_1/RestoreV2:35*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	<
ź
save_1/Assign_36Assignvc/dense/kernel/Adamsave_1/RestoreV2:36*
validate_shape(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
use_locking(*
T0
ž
save_1/Assign_37Assignvc/dense/kernel/Adam_1save_1/RestoreV2:37*
T0*
use_locking(*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@vc/dense/kernel
ł
save_1/Assign_38Assignvc/dense_1/biassave_1/RestoreV2:38*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias
¸
save_1/Assign_39Assignvc/dense_1/bias/Adamsave_1/RestoreV2:39*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(*"
_class
loc:@vc/dense_1/bias
ş
save_1/Assign_40Assignvc/dense_1/bias/Adam_1save_1/RestoreV2:40*
_output_shapes	
:*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(
ź
save_1/Assign_41Assignvc/dense_1/kernelsave_1/RestoreV2:41* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
T0
Á
save_1/Assign_42Assignvc/dense_1/kernel/Adamsave_1/RestoreV2:42*
T0*
use_locking(* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
Ă
save_1/Assign_43Assignvc/dense_1/kernel/Adam_1save_1/RestoreV2:43* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
˛
save_1/Assign_44Assignvc/dense_2/biassave_1/RestoreV2:44*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0*
use_locking(*
validate_shape(
ˇ
save_1/Assign_45Assignvc/dense_2/bias/Adamsave_1/RestoreV2:45*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@vc/dense_2/bias
š
save_1/Assign_46Assignvc/dense_2/bias/Adam_1save_1/RestoreV2:46*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
ť
save_1/Assign_47Assignvc/dense_2/kernelsave_1/RestoreV2:47*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0*$
_class
loc:@vc/dense_2/kernel
Ŕ
save_1/Assign_48Assignvc/dense_2/kernel/Adamsave_1/RestoreV2:48*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
Â
save_1/Assign_49Assignvc/dense_2/kernel/Adam_1save_1/RestoreV2:49*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel
Ż
save_1/Assign_50Assignvf/dense/biassave_1/RestoreV2:50*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vf/dense/bias
´
save_1/Assign_51Assignvf/dense/bias/Adamsave_1/RestoreV2:51*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(
ś
save_1/Assign_52Assignvf/dense/bias/Adam_1save_1/RestoreV2:52*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
validate_shape(*
use_locking(
ˇ
save_1/Assign_53Assignvf/dense/kernelsave_1/RestoreV2:53*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
use_locking(
ź
save_1/Assign_54Assignvf/dense/kernel/Adamsave_1/RestoreV2:54*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	<
ž
save_1/Assign_55Assignvf/dense/kernel/Adam_1save_1/RestoreV2:55*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
use_locking(*
validate_shape(
ł
save_1/Assign_56Assignvf/dense_1/biassave_1/RestoreV2:56*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
¸
save_1/Assign_57Assignvf/dense_1/bias/Adamsave_1/RestoreV2:57*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias
ş
save_1/Assign_58Assignvf/dense_1/bias/Adam_1save_1/RestoreV2:58*
use_locking(*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0
ź
save_1/Assign_59Assignvf/dense_1/kernelsave_1/RestoreV2:59*
T0*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
validate_shape(
Á
save_1/Assign_60Assignvf/dense_1/kernel/Adamsave_1/RestoreV2:60*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:

Ă
save_1/Assign_61Assignvf/dense_1/kernel/Adam_1save_1/RestoreV2:61*
validate_shape(*
T0* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vf/dense_1/kernel
˛
save_1/Assign_62Assignvf/dense_2/biassave_1/RestoreV2:62*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_2/bias
ˇ
save_1/Assign_63Assignvf/dense_2/bias/Adamsave_1/RestoreV2:63*"
_class
loc:@vf/dense_2/bias*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
š
save_1/Assign_64Assignvf/dense_2/bias/Adam_1save_1/RestoreV2:64*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias
ť
save_1/Assign_65Assignvf/dense_2/kernelsave_1/RestoreV2:65*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
Ŕ
save_1/Assign_66Assignvf/dense_2/kernel/Adamsave_1/RestoreV2:66*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(
Â
save_1/Assign_67Assignvf/dense_2/kernel/Adam_1save_1/RestoreV2:67*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	


save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
shape: *
dtype0*
_output_shapes
: 

save_2/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_e6916dc3e25d432cbc818bc3c09d1fe3/part
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
S
save_2/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_2/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0

save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
÷
save_2/SaveV2/tensor_namesConst*
_output_shapes
:D*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
đ
save_2/SaveV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D*
dtype0
Ň
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*)
_class
loc:@save_2/ShardedFilename*
T0*
_output_shapes
: 
Ł
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(

save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
_output_shapes
: *
T0
ú
save_2/RestoreV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
ó
!save_2/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ę
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ş
save_2/AssignAssignbeta1_powersave_2/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
T0
°
save_2/Assign_1Assignbeta1_power_1save_2/RestoreV2:1*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
validate_shape(
¨
save_2/Assign_2Assignbeta1_power_2save_2/RestoreV2:2*
_output_shapes
: *
use_locking(*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias
Ž
save_2/Assign_3Assignbeta2_powersave_2/RestoreV2:3*
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: 
°
save_2/Assign_4Assignbeta2_power_1save_2/RestoreV2:4*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(
¨
save_2/Assign_5Assignbeta2_power_2save_2/RestoreV2:5*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
T0
¸
save_2/Assign_6Assignpenalty/penalty_paramsave_2/RestoreV2:6*
validate_shape(*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(
˝
save_2/Assign_7Assignpenalty/penalty_param/Adamsave_2/RestoreV2:7*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
validate_shape(*
T0
ż
save_2/Assign_8Assignpenalty/penalty_param/Adam_1save_2/RestoreV2:8*
validate_shape(*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param*
T0
ż
save_2/Assign_9Assignpenalty/penalty_param/Adam_2save_2/RestoreV2:9*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0*
validate_shape(
Á
save_2/Assign_10Assignpenalty/penalty_param/Adam_3save_2/RestoreV2:10*
use_locking(*
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Ż
save_2/Assign_11Assignpi/dense/biassave_2/RestoreV2:11* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
´
save_2/Assign_12Assignpi/dense/bias/Adamsave_2/RestoreV2:12*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(* 
_class
loc:@pi/dense/bias
ś
save_2/Assign_13Assignpi/dense/bias/Adam_1save_2/RestoreV2:13*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
T0
ˇ
save_2/Assign_14Assignpi/dense/kernelsave_2/RestoreV2:14*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	<
ź
save_2/Assign_15Assignpi/dense/kernel/Adamsave_2/RestoreV2:15*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(
ž
save_2/Assign_16Assignpi/dense/kernel/Adam_1save_2/RestoreV2:16*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
use_locking(
ł
save_2/Assign_17Assignpi/dense_1/biassave_2/RestoreV2:17*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:
¸
save_2/Assign_18Assignpi/dense_1/bias/Adamsave_2/RestoreV2:18*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias
ş
save_2/Assign_19Assignpi/dense_1/bias/Adam_1save_2/RestoreV2:19*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0
ź
save_2/Assign_20Assignpi/dense_1/kernelsave_2/RestoreV2:20*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

Á
save_2/Assign_21Assignpi/dense_1/kernel/Adamsave_2/RestoreV2:21* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(*
T0
Ă
save_2/Assign_22Assignpi/dense_1/kernel/Adam_1save_2/RestoreV2:22*
validate_shape(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(
˛
save_2/Assign_23Assignpi/dense_2/biassave_2/RestoreV2:23*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0*
validate_shape(
ˇ
save_2/Assign_24Assignpi/dense_2/bias/Adamsave_2/RestoreV2:24*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:
š
save_2/Assign_25Assignpi/dense_2/bias/Adam_1save_2/RestoreV2:25*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:
ť
save_2/Assign_26Assignpi/dense_2/kernelsave_2/RestoreV2:26*
T0*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(
Ŕ
save_2/Assign_27Assignpi/dense_2/kernel/Adamsave_2/RestoreV2:27*
_output_shapes
:	*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0
Â
save_2/Assign_28Assignpi/dense_2/kernel/Adam_1save_2/RestoreV2:28*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
:	*
validate_shape(
¨
save_2/Assign_29Assign
pi/log_stdsave_2/RestoreV2:29*
use_locking(*
_output_shapes
:*
T0*
_class
loc:@pi/log_std*
validate_shape(
­
save_2/Assign_30Assignpi/log_std/Adamsave_2/RestoreV2:30*
_class
loc:@pi/log_std*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
Ż
save_2/Assign_31Assignpi/log_std/Adam_1save_2/RestoreV2:31*
validate_shape(*
use_locking(*
_class
loc:@pi/log_std*
_output_shapes
:*
T0
Ż
save_2/Assign_32Assignvc/dense/biassave_2/RestoreV2:32*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(* 
_class
loc:@vc/dense/bias
´
save_2/Assign_33Assignvc/dense/bias/Adamsave_2/RestoreV2:33* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
ś
save_2/Assign_34Assignvc/dense/bias/Adam_1save_2/RestoreV2:34*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias
ˇ
save_2/Assign_35Assignvc/dense/kernelsave_2/RestoreV2:35*
_output_shapes
:	<*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense/kernel*
T0
ź
save_2/Assign_36Assignvc/dense/kernel/Adamsave_2/RestoreV2:36*
T0*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(
ž
save_2/Assign_37Assignvc/dense/kernel/Adam_1save_2/RestoreV2:37*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0*
_output_shapes
:	<*
validate_shape(
ł
save_2/Assign_38Assignvc/dense_1/biassave_2/RestoreV2:38*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
validate_shape(
¸
save_2/Assign_39Assignvc/dense_1/bias/Adamsave_2/RestoreV2:39*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(
ş
save_2/Assign_40Assignvc/dense_1/bias/Adam_1save_2/RestoreV2:40*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias
ź
save_2/Assign_41Assignvc/dense_1/kernelsave_2/RestoreV2:41* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(
Á
save_2/Assign_42Assignvc/dense_1/kernel/Adamsave_2/RestoreV2:42*
T0*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
Ă
save_2/Assign_43Assignvc/dense_1/kernel/Adam_1save_2/RestoreV2:43*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking(
˛
save_2/Assign_44Assignvc/dense_2/biassave_2/RestoreV2:44*
use_locking(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
ˇ
save_2/Assign_45Assignvc/dense_2/bias/Adamsave_2/RestoreV2:45*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias
š
save_2/Assign_46Assignvc/dense_2/bias/Adam_1save_2/RestoreV2:46*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:
ť
save_2/Assign_47Assignvc/dense_2/kernelsave_2/RestoreV2:47*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0
Ŕ
save_2/Assign_48Assignvc/dense_2/kernel/Adamsave_2/RestoreV2:48*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
Â
save_2/Assign_49Assignvc/dense_2/kernel/Adam_1save_2/RestoreV2:49*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	*
T0*
validate_shape(
Ż
save_2/Assign_50Assignvf/dense/biassave_2/RestoreV2:50* 
_class
loc:@vf/dense/bias*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(
´
save_2/Assign_51Assignvf/dense/bias/Adamsave_2/RestoreV2:51*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0
ś
save_2/Assign_52Assignvf/dense/bias/Adam_1save_2/RestoreV2:52*
use_locking(*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(
ˇ
save_2/Assign_53Assignvf/dense/kernelsave_2/RestoreV2:53*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
use_locking(*
T0*
validate_shape(
ź
save_2/Assign_54Assignvf/dense/kernel/Adamsave_2/RestoreV2:54*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
use_locking(
ž
save_2/Assign_55Assignvf/dense/kernel/Adam_1save_2/RestoreV2:55*"
_class
loc:@vf/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	<
ł
save_2/Assign_56Assignvf/dense_1/biassave_2/RestoreV2:56*
validate_shape(*
_output_shapes	
:*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0
¸
save_2/Assign_57Assignvf/dense_1/bias/Adamsave_2/RestoreV2:57*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:
ş
save_2/Assign_58Assignvf/dense_1/bias/Adam_1save_2/RestoreV2:58*
_output_shapes	
:*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0
ź
save_2/Assign_59Assignvf/dense_1/kernelsave_2/RestoreV2:59*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
T0
Á
save_2/Assign_60Assignvf/dense_1/kernel/Adamsave_2/RestoreV2:60*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0
Ă
save_2/Assign_61Assignvf/dense_1/kernel/Adam_1save_2/RestoreV2:61*
validate_shape(* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
T0
˛
save_2/Assign_62Assignvf/dense_2/biassave_2/RestoreV2:62*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:
ˇ
save_2/Assign_63Assignvf/dense_2/bias/Adamsave_2/RestoreV2:63*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias
š
save_2/Assign_64Assignvf/dense_2/bias/Adam_1save_2/RestoreV2:64*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:*
use_locking(
ť
save_2/Assign_65Assignvf/dense_2/kernelsave_2/RestoreV2:65*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(
Ŕ
save_2/Assign_66Assignvf/dense_2/kernel/Adamsave_2/RestoreV2:66*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	
Â
save_2/Assign_67Assignvf/dense_2/kernel/Adam_1save_2/RestoreV2:67*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	


save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_40^save_2/Assign_41^save_2/Assign_42^save_2/Assign_43^save_2/Assign_44^save_2/Assign_45^save_2/Assign_46^save_2/Assign_47^save_2/Assign_48^save_2/Assign_49^save_2/Assign_5^save_2/Assign_50^save_2/Assign_51^save_2/Assign_52^save_2/Assign_53^save_2/Assign_54^save_2/Assign_55^save_2/Assign_56^save_2/Assign_57^save_2/Assign_58^save_2/Assign_59^save_2/Assign_6^save_2/Assign_60^save_2/Assign_61^save_2/Assign_62^save_2/Assign_63^save_2/Assign_64^save_2/Assign_65^save_2/Assign_66^save_2/Assign_67^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
_output_shapes
: *
shape: *
dtype0

save_3/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_65d9dba442c84c74ae8cb8a8adf9811a/part*
dtype0
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
S
save_3/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_3/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 

save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
÷
save_3/SaveV2/tensor_namesConst*
_output_shapes
:D*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
đ
save_3/SaveV2/shape_and_slicesConst*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D
Ň
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
_output_shapes
: *)
_class
loc:@save_3/ShardedFilename*
T0
Ł
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*

axis *
N*
T0*
_output_shapes
:

save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(

save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
_output_shapes
: *
T0
ú
save_3/RestoreV2/tensor_namesConst*
_output_shapes
:D*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ó
!save_3/RestoreV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ę
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ş
save_3/AssignAssignbeta1_powersave_3/RestoreV2*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: 
°
save_3/Assign_1Assignbeta1_power_1save_3/RestoreV2:1*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: 
¨
save_3/Assign_2Assignbeta1_power_2save_3/RestoreV2:2*
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0*
use_locking(
Ž
save_3/Assign_3Assignbeta2_powersave_3/RestoreV2:3*
_output_shapes
: *
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0
°
save_3/Assign_4Assignbeta2_power_1save_3/RestoreV2:4*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(
¨
save_3/Assign_5Assignbeta2_power_2save_3/RestoreV2:5*
_output_shapes
: *
T0*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias
¸
save_3/Assign_6Assignpenalty/penalty_paramsave_3/RestoreV2:6*
_output_shapes
: *
use_locking(*
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param
˝
save_3/Assign_7Assignpenalty/penalty_param/Adamsave_3/RestoreV2:7*(
_class
loc:@penalty/penalty_param*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
ż
save_3/Assign_8Assignpenalty/penalty_param/Adam_1save_3/RestoreV2:8*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
ż
save_3/Assign_9Assignpenalty/penalty_param/Adam_2save_3/RestoreV2:9*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
_output_shapes
: *
T0
Á
save_3/Assign_10Assignpenalty/penalty_param/Adam_3save_3/RestoreV2:10*
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0
Ż
save_3/Assign_11Assignpi/dense/biassave_3/RestoreV2:11*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
T0
´
save_3/Assign_12Assignpi/dense/bias/Adamsave_3/RestoreV2:12*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
use_locking(
ś
save_3/Assign_13Assignpi/dense/bias/Adam_1save_3/RestoreV2:13*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes	
:*
T0
ˇ
save_3/Assign_14Assignpi/dense/kernelsave_3/RestoreV2:14*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	<
ź
save_3/Assign_15Assignpi/dense/kernel/Adamsave_3/RestoreV2:15*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0*
validate_shape(
ž
save_3/Assign_16Assignpi/dense/kernel/Adam_1save_3/RestoreV2:16*
_output_shapes
:	<*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(
ł
save_3/Assign_17Assignpi/dense_1/biassave_3/RestoreV2:17*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias
¸
save_3/Assign_18Assignpi/dense_1/bias/Adamsave_3/RestoreV2:18*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias
ş
save_3/Assign_19Assignpi/dense_1/bias/Adam_1save_3/RestoreV2:19*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias
ź
save_3/Assign_20Assignpi/dense_1/kernelsave_3/RestoreV2:20*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
T0*
use_locking(
Á
save_3/Assign_21Assignpi/dense_1/kernel/Adamsave_3/RestoreV2:21*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:

Ă
save_3/Assign_22Assignpi/dense_1/kernel/Adam_1save_3/RestoreV2:22*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

˛
save_3/Assign_23Assignpi/dense_2/biassave_3/RestoreV2:23*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
ˇ
save_3/Assign_24Assignpi/dense_2/bias/Adamsave_3/RestoreV2:24*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(
š
save_3/Assign_25Assignpi/dense_2/bias/Adam_1save_3/RestoreV2:25*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
ť
save_3/Assign_26Assignpi/dense_2/kernelsave_3/RestoreV2:26*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
T0*
use_locking(*
validate_shape(
Ŕ
save_3/Assign_27Assignpi/dense_2/kernel/Adamsave_3/RestoreV2:27*
T0*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(
Â
save_3/Assign_28Assignpi/dense_2/kernel/Adam_1save_3/RestoreV2:28*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
:	
¨
save_3/Assign_29Assign
pi/log_stdsave_3/RestoreV2:29*
_class
loc:@pi/log_std*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
­
save_3/Assign_30Assignpi/log_std/Adamsave_3/RestoreV2:30*
validate_shape(*
use_locking(*
_class
loc:@pi/log_std*
_output_shapes
:*
T0
Ż
save_3/Assign_31Assignpi/log_std/Adam_1save_3/RestoreV2:31*
_output_shapes
:*
_class
loc:@pi/log_std*
use_locking(*
validate_shape(*
T0
Ż
save_3/Assign_32Assignvc/dense/biassave_3/RestoreV2:32*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:
´
save_3/Assign_33Assignvc/dense/bias/Adamsave_3/RestoreV2:33*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0
ś
save_3/Assign_34Assignvc/dense/bias/Adam_1save_3/RestoreV2:34*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0*
validate_shape(
ˇ
save_3/Assign_35Assignvc/dense/kernelsave_3/RestoreV2:35*
_output_shapes
:	<*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(
ź
save_3/Assign_36Assignvc/dense/kernel/Adamsave_3/RestoreV2:36*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(*
_output_shapes
:	<
ž
save_3/Assign_37Assignvc/dense/kernel/Adam_1save_3/RestoreV2:37*
T0*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
use_locking(
ł
save_3/Assign_38Assignvc/dense_1/biassave_3/RestoreV2:38*
use_locking(*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:
¸
save_3/Assign_39Assignvc/dense_1/bias/Adamsave_3/RestoreV2:39*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(
ş
save_3/Assign_40Assignvc/dense_1/bias/Adam_1save_3/RestoreV2:40*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias
ź
save_3/Assign_41Assignvc/dense_1/kernelsave_3/RestoreV2:41*
use_locking(* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
Á
save_3/Assign_42Assignvc/dense_1/kernel/Adamsave_3/RestoreV2:42*
use_locking(* 
_output_shapes
:
*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
Ă
save_3/Assign_43Assignvc/dense_1/kernel/Adam_1save_3/RestoreV2:43* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
validate_shape(*
T0
˛
save_3/Assign_44Assignvc/dense_2/biassave_3/RestoreV2:44*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(
ˇ
save_3/Assign_45Assignvc/dense_2/bias/Adamsave_3/RestoreV2:45*"
_class
loc:@vc/dense_2/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
š
save_3/Assign_46Assignvc/dense_2/bias/Adam_1save_3/RestoreV2:46*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(
ť
save_3/Assign_47Assignvc/dense_2/kernelsave_3/RestoreV2:47*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	
Ŕ
save_3/Assign_48Assignvc/dense_2/kernel/Adamsave_3/RestoreV2:48*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(
Â
save_3/Assign_49Assignvc/dense_2/kernel/Adam_1save_3/RestoreV2:49*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	
Ż
save_3/Assign_50Assignvf/dense/biassave_3/RestoreV2:50*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
´
save_3/Assign_51Assignvf/dense/bias/Adamsave_3/RestoreV2:51*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
ś
save_3/Assign_52Assignvf/dense/bias/Adam_1save_3/RestoreV2:52* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0
ˇ
save_3/Assign_53Assignvf/dense/kernelsave_3/RestoreV2:53*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(
ź
save_3/Assign_54Assignvf/dense/kernel/Adamsave_3/RestoreV2:54*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
validate_shape(
ž
save_3/Assign_55Assignvf/dense/kernel/Adam_1save_3/RestoreV2:55*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
validate_shape(*
use_locking(*
T0
ł
save_3/Assign_56Assignvf/dense_1/biassave_3/RestoreV2:56*
_output_shapes	
:*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0*
validate_shape(
¸
save_3/Assign_57Assignvf/dense_1/bias/Adamsave_3/RestoreV2:57*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
ş
save_3/Assign_58Assignvf/dense_1/bias/Adam_1save_3/RestoreV2:58*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(
ź
save_3/Assign_59Assignvf/dense_1/kernelsave_3/RestoreV2:59*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel
Á
save_3/Assign_60Assignvf/dense_1/kernel/Adamsave_3/RestoreV2:60*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:

Ă
save_3/Assign_61Assignvf/dense_1/kernel/Adam_1save_3/RestoreV2:61*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
validate_shape(*
T0
˛
save_3/Assign_62Assignvf/dense_2/biassave_3/RestoreV2:62*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
ˇ
save_3/Assign_63Assignvf/dense_2/bias/Adamsave_3/RestoreV2:63*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
validate_shape(
š
save_3/Assign_64Assignvf/dense_2/bias/Adam_1save_3/RestoreV2:64*"
_class
loc:@vf/dense_2/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
ť
save_3/Assign_65Assignvf/dense_2/kernelsave_3/RestoreV2:65*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(
Ŕ
save_3/Assign_66Assignvf/dense_2/kernel/Adamsave_3/RestoreV2:66*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	
Â
save_3/Assign_67Assignvf/dense_2/kernel/Adam_1save_3/RestoreV2:67*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(


save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_45^save_3/Assign_46^save_3/Assign_47^save_3/Assign_48^save_3/Assign_49^save_3/Assign_5^save_3/Assign_50^save_3/Assign_51^save_3/Assign_52^save_3/Assign_53^save_3/Assign_54^save_3/Assign_55^save_3/Assign_56^save_3/Assign_57^save_3/Assign_58^save_3/Assign_59^save_3/Assign_6^save_3/Assign_60^save_3/Assign_61^save_3/Assign_62^save_3/Assign_63^save_3/Assign_64^save_3/Assign_65^save_3/Assign_66^save_3/Assign_67^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard
[
save_4/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
shape: *
_output_shapes
: *
dtype0

save_4/StringJoin/inputs_1Const*<
value3B1 B+_temp_79bb840cf5004a8b841b45ec8a9f3e72/part*
_output_shapes
: *
dtype0
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
S
save_4/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_4/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
÷
save_4/SaveV2/tensor_namesConst*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
đ
save_4/SaveV2/shape_and_slicesConst*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D
Ň
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_4/ShardedFilename
Ł
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*

axis *
_output_shapes
:*
N*
T0

save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(

save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
T0*
_output_shapes
: 
ú
save_4/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:D
ó
!save_4/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ę
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*R
dtypesH
F2D*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ş
save_4/AssignAssignbeta1_powersave_4/RestoreV2*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
°
save_4/Assign_1Assignbeta1_power_1save_4/RestoreV2:1*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
¨
save_4/Assign_2Assignbeta1_power_2save_4/RestoreV2:2*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
use_locking(*
T0
Ž
save_4/Assign_3Assignbeta2_powersave_4/RestoreV2:3*
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(
°
save_4/Assign_4Assignbeta2_power_1save_4/RestoreV2:4*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(
¨
save_4/Assign_5Assignbeta2_power_2save_4/RestoreV2:5*
use_locking(*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes
: 
¸
save_4/Assign_6Assignpenalty/penalty_paramsave_4/RestoreV2:6*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: 
˝
save_4/Assign_7Assignpenalty/penalty_param/Adamsave_4/RestoreV2:7*
T0*
_output_shapes
: *
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param
ż
save_4/Assign_8Assignpenalty/penalty_param/Adam_1save_4/RestoreV2:8*(
_class
loc:@penalty/penalty_param*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
ż
save_4/Assign_9Assignpenalty/penalty_param/Adam_2save_4/RestoreV2:9*
_output_shapes
: *
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0
Á
save_4/Assign_10Assignpenalty/penalty_param/Adam_3save_4/RestoreV2:10*
T0*
validate_shape(*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
Ż
save_4/Assign_11Assignpi/dense/biassave_4/RestoreV2:11*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias
´
save_4/Assign_12Assignpi/dense/bias/Adamsave_4/RestoreV2:12*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0*
use_locking(
ś
save_4/Assign_13Assignpi/dense/bias/Adam_1save_4/RestoreV2:13*
validate_shape(*
_output_shapes	
:*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
ˇ
save_4/Assign_14Assignpi/dense/kernelsave_4/RestoreV2:14*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
T0
ź
save_4/Assign_15Assignpi/dense/kernel/Adamsave_4/RestoreV2:15*
validate_shape(*
T0*
_output_shapes
:	<*
use_locking(*"
_class
loc:@pi/dense/kernel
ž
save_4/Assign_16Assignpi/dense/kernel/Adam_1save_4/RestoreV2:16*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(
ł
save_4/Assign_17Assignpi/dense_1/biassave_4/RestoreV2:17*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias
¸
save_4/Assign_18Assignpi/dense_1/bias/Adamsave_4/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_4/Assign_19Assignpi/dense_1/bias/Adam_1save_4/RestoreV2:19*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias
ź
save_4/Assign_20Assignpi/dense_1/kernelsave_4/RestoreV2:20*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0
Á
save_4/Assign_21Assignpi/dense_1/kernel/Adamsave_4/RestoreV2:21*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

Ă
save_4/Assign_22Assignpi/dense_1/kernel/Adam_1save_4/RestoreV2:22*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:
*
validate_shape(*
use_locking(
˛
save_4/Assign_23Assignpi/dense_2/biassave_4/RestoreV2:23*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(
ˇ
save_4/Assign_24Assignpi/dense_2/bias/Adamsave_4/RestoreV2:24*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
š
save_4/Assign_25Assignpi/dense_2/bias/Adam_1save_4/RestoreV2:25*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0
ť
save_4/Assign_26Assignpi/dense_2/kernelsave_4/RestoreV2:26*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel
Ŕ
save_4/Assign_27Assignpi/dense_2/kernel/Adamsave_4/RestoreV2:27*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(
Â
save_4/Assign_28Assignpi/dense_2/kernel/Adam_1save_4/RestoreV2:28*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	*
validate_shape(
¨
save_4/Assign_29Assign
pi/log_stdsave_4/RestoreV2:29*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(
­
save_4/Assign_30Assignpi/log_std/Adamsave_4/RestoreV2:30*
validate_shape(*
_class
loc:@pi/log_std*
T0*
use_locking(*
_output_shapes
:
Ż
save_4/Assign_31Assignpi/log_std/Adam_1save_4/RestoreV2:31*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
Ż
save_4/Assign_32Assignvc/dense/biassave_4/RestoreV2:32*
_output_shapes	
:*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(
´
save_4/Assign_33Assignvc/dense/bias/Adamsave_4/RestoreV2:33*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0*
use_locking(
ś
save_4/Assign_34Assignvc/dense/bias/Adam_1save_4/RestoreV2:34*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0
ˇ
save_4/Assign_35Assignvc/dense/kernelsave_4/RestoreV2:35*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
validate_shape(*
T0
ź
save_4/Assign_36Assignvc/dense/kernel/Adamsave_4/RestoreV2:36*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel
ž
save_4/Assign_37Assignvc/dense/kernel/Adam_1save_4/RestoreV2:37*
T0*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(
ł
save_4/Assign_38Assignvc/dense_1/biassave_4/RestoreV2:38*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:
¸
save_4/Assign_39Assignvc/dense_1/bias/Adamsave_4/RestoreV2:39*
validate_shape(*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
use_locking(
ş
save_4/Assign_40Assignvc/dense_1/bias/Adam_1save_4/RestoreV2:40*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vc/dense_1/bias
ź
save_4/Assign_41Assignvc/dense_1/kernelsave_4/RestoreV2:41*
T0*
validate_shape(* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vc/dense_1/kernel
Á
save_4/Assign_42Assignvc/dense_1/kernel/Adamsave_4/RestoreV2:42*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:

Ă
save_4/Assign_43Assignvc/dense_1/kernel/Adam_1save_4/RestoreV2:43*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:

˛
save_4/Assign_44Assignvc/dense_2/biassave_4/RestoreV2:44*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
ˇ
save_4/Assign_45Assignvc/dense_2/bias/Adamsave_4/RestoreV2:45*
T0*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_4/Assign_46Assignvc/dense_2/bias/Adam_1save_4/RestoreV2:46*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
ť
save_4/Assign_47Assignvc/dense_2/kernelsave_4/RestoreV2:47*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0
Ŕ
save_4/Assign_48Assignvc/dense_2/kernel/Adamsave_4/RestoreV2:48*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
T0*
use_locking(
Â
save_4/Assign_49Assignvc/dense_2/kernel/Adam_1save_4/RestoreV2:49*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ż
save_4/Assign_50Assignvf/dense/biassave_4/RestoreV2:50*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
´
save_4/Assign_51Assignvf/dense/bias/Adamsave_4/RestoreV2:51* 
_class
loc:@vf/dense/bias*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(
ś
save_4/Assign_52Assignvf/dense/bias/Adam_1save_4/RestoreV2:52*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias
ˇ
save_4/Assign_53Assignvf/dense/kernelsave_4/RestoreV2:53*"
_class
loc:@vf/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	<*
use_locking(
ź
save_4/Assign_54Assignvf/dense/kernel/Adamsave_4/RestoreV2:54*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	<
ž
save_4/Assign_55Assignvf/dense/kernel/Adam_1save_4/RestoreV2:55*
T0*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(
ł
save_4/Assign_56Assignvf/dense_1/biassave_4/RestoreV2:56*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(*"
_class
loc:@vf/dense_1/bias
¸
save_4/Assign_57Assignvf/dense_1/bias/Adamsave_4/RestoreV2:57*
_output_shapes	
:*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0*
validate_shape(
ş
save_4/Assign_58Assignvf/dense_1/bias/Adam_1save_4/RestoreV2:58*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(
ź
save_4/Assign_59Assignvf/dense_1/kernelsave_4/RestoreV2:59*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:

Á
save_4/Assign_60Assignvf/dense_1/kernel/Adamsave_4/RestoreV2:60*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
use_locking(
Ă
save_4/Assign_61Assignvf/dense_1/kernel/Adam_1save_4/RestoreV2:61*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:

˛
save_4/Assign_62Assignvf/dense_2/biassave_4/RestoreV2:62*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
ˇ
save_4/Assign_63Assignvf/dense_2/bias/Adamsave_4/RestoreV2:63*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0*
use_locking(
š
save_4/Assign_64Assignvf/dense_2/bias/Adam_1save_4/RestoreV2:64*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:*
T0
ť
save_4/Assign_65Assignvf/dense_2/kernelsave_4/RestoreV2:65*
use_locking(*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0
Ŕ
save_4/Assign_66Assignvf/dense_2/kernel/Adamsave_4/RestoreV2:66*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(
Â
save_4/Assign_67Assignvf/dense_2/kernel/Adam_1save_4/RestoreV2:67*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	


save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_2^save_4/Assign_20^save_4/Assign_21^save_4/Assign_22^save_4/Assign_23^save_4/Assign_24^save_4/Assign_25^save_4/Assign_26^save_4/Assign_27^save_4/Assign_28^save_4/Assign_29^save_4/Assign_3^save_4/Assign_30^save_4/Assign_31^save_4/Assign_32^save_4/Assign_33^save_4/Assign_34^save_4/Assign_35^save_4/Assign_36^save_4/Assign_37^save_4/Assign_38^save_4/Assign_39^save_4/Assign_4^save_4/Assign_40^save_4/Assign_41^save_4/Assign_42^save_4/Assign_43^save_4/Assign_44^save_4/Assign_45^save_4/Assign_46^save_4/Assign_47^save_4/Assign_48^save_4/Assign_49^save_4/Assign_5^save_4/Assign_50^save_4/Assign_51^save_4/Assign_52^save_4/Assign_53^save_4/Assign_54^save_4/Assign_55^save_4/Assign_56^save_4/Assign_57^save_4/Assign_58^save_4/Assign_59^save_4/Assign_6^save_4/Assign_60^save_4/Assign_61^save_4/Assign_62^save_4/Assign_63^save_4/Assign_64^save_4/Assign_65^save_4/Assign_66^save_4/Assign_67^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard
[
save_5/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
_output_shapes
: *
dtype0*
shape: 

save_5/StringJoin/inputs_1Const*<
value3B1 B+_temp_387d9355c7c5485dbd82c0defd1624b5/part*
_output_shapes
: *
dtype0
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
save_5/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
÷
save_5/SaveV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D*
dtype0
đ
save_5/SaveV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:D
Ň
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_5/ShardedFilename
Ł
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*
_output_shapes
:*
N*
T0*

axis 

save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(

save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
T0*
_output_shapes
: 
ú
save_5/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:D
ó
!save_5/RestoreV2/shape_and_slicesConst*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D
ę
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ş
save_5/AssignAssignbeta1_powersave_5/RestoreV2*
validate_shape(*
T0*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
°
save_5/Assign_1Assignbeta1_power_1save_5/RestoreV2:1*
_output_shapes
: *
T0*
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(
¨
save_5/Assign_2Assignbeta1_power_2save_5/RestoreV2:2*
_output_shapes
: *
validate_shape(*
use_locking(*
T0* 
_class
loc:@vc/dense/bias
Ž
save_5/Assign_3Assignbeta2_powersave_5/RestoreV2:3*
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(
°
save_5/Assign_4Assignbeta2_power_1save_5/RestoreV2:4*
_output_shapes
: *
use_locking(*
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0
¨
save_5/Assign_5Assignbeta2_power_2save_5/RestoreV2:5*
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
use_locking(*
T0
¸
save_5/Assign_6Assignpenalty/penalty_paramsave_5/RestoreV2:6*
use_locking(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
validate_shape(
˝
save_5/Assign_7Assignpenalty/penalty_param/Adamsave_5/RestoreV2:7*
validate_shape(*
use_locking(*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
ż
save_5/Assign_8Assignpenalty/penalty_param/Adam_1save_5/RestoreV2:8*
T0*
validate_shape(*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param
ż
save_5/Assign_9Assignpenalty/penalty_param/Adam_2save_5/RestoreV2:9*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(*
use_locking(
Á
save_5/Assign_10Assignpenalty/penalty_param/Adam_3save_5/RestoreV2:10*(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(*
_output_shapes
: *
T0
Ż
save_5/Assign_11Assignpi/dense/biassave_5/RestoreV2:11*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
T0
´
save_5/Assign_12Assignpi/dense/bias/Adamsave_5/RestoreV2:12*
_output_shapes	
:*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(
ś
save_5/Assign_13Assignpi/dense/bias/Adam_1save_5/RestoreV2:13*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0*
use_locking(
ˇ
save_5/Assign_14Assignpi/dense/kernelsave_5/RestoreV2:14*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	<*
T0
ź
save_5/Assign_15Assignpi/dense/kernel/Adamsave_5/RestoreV2:15*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
use_locking(
ž
save_5/Assign_16Assignpi/dense/kernel/Adam_1save_5/RestoreV2:16*
_output_shapes
:	<*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0
ł
save_5/Assign_17Assignpi/dense_1/biassave_5/RestoreV2:17*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(*
T0
¸
save_5/Assign_18Assignpi/dense_1/bias/Adamsave_5/RestoreV2:18*
T0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(
ş
save_5/Assign_19Assignpi/dense_1/bias/Adam_1save_5/RestoreV2:19*
T0*
use_locking(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
validate_shape(
ź
save_5/Assign_20Assignpi/dense_1/kernelsave_5/RestoreV2:20*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel
Á
save_5/Assign_21Assignpi/dense_1/kernel/Adamsave_5/RestoreV2:21*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:
*
validate_shape(
Ă
save_5/Assign_22Assignpi/dense_1/kernel/Adam_1save_5/RestoreV2:22*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

˛
save_5/Assign_23Assignpi/dense_2/biassave_5/RestoreV2:23*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:
ˇ
save_5/Assign_24Assignpi/dense_2/bias/Adamsave_5/RestoreV2:24*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
š
save_5/Assign_25Assignpi/dense_2/bias/Adam_1save_5/RestoreV2:25*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
ť
save_5/Assign_26Assignpi/dense_2/kernelsave_5/RestoreV2:26*
_output_shapes
:	*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(
Ŕ
save_5/Assign_27Assignpi/dense_2/kernel/Adamsave_5/RestoreV2:27*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	
Â
save_5/Assign_28Assignpi/dense_2/kernel/Adam_1save_5/RestoreV2:28*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(
¨
save_5/Assign_29Assign
pi/log_stdsave_5/RestoreV2:29*
T0*
_output_shapes
:*
_class
loc:@pi/log_std*
use_locking(*
validate_shape(
­
save_5/Assign_30Assignpi/log_std/Adamsave_5/RestoreV2:30*
validate_shape(*
_output_shapes
:*
T0*
_class
loc:@pi/log_std*
use_locking(
Ż
save_5/Assign_31Assignpi/log_std/Adam_1save_5/RestoreV2:31*
validate_shape(*
_class
loc:@pi/log_std*
use_locking(*
T0*
_output_shapes
:
Ż
save_5/Assign_32Assignvc/dense/biassave_5/RestoreV2:32*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
use_locking(
´
save_5/Assign_33Assignvc/dense/bias/Adamsave_5/RestoreV2:33*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias
ś
save_5/Assign_34Assignvc/dense/bias/Adam_1save_5/RestoreV2:34*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
use_locking(
ˇ
save_5/Assign_35Assignvc/dense/kernelsave_5/RestoreV2:35*
validate_shape(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<*
use_locking(
ź
save_5/Assign_36Assignvc/dense/kernel/Adamsave_5/RestoreV2:36*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<
ž
save_5/Assign_37Assignvc/dense/kernel/Adam_1save_5/RestoreV2:37*
use_locking(*
_output_shapes
:	<*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel
ł
save_5/Assign_38Assignvc/dense_1/biassave_5/RestoreV2:38*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
¸
save_5/Assign_39Assignvc/dense_1/bias/Adamsave_5/RestoreV2:39*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0
ş
save_5/Assign_40Assignvc/dense_1/bias/Adam_1save_5/RestoreV2:40*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(
ź
save_5/Assign_41Assignvc/dense_1/kernelsave_5/RestoreV2:41*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:

Á
save_5/Assign_42Assignvc/dense_1/kernel/Adamsave_5/RestoreV2:42*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
Ă
save_5/Assign_43Assignvc/dense_1/kernel/Adam_1save_5/RestoreV2:43*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0
˛
save_5/Assign_44Assignvc/dense_2/biassave_5/RestoreV2:44*"
_class
loc:@vc/dense_2/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
ˇ
save_5/Assign_45Assignvc/dense_2/bias/Adamsave_5/RestoreV2:45*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(
š
save_5/Assign_46Assignvc/dense_2/bias/Adam_1save_5/RestoreV2:46*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:
ť
save_5/Assign_47Assignvc/dense_2/kernelsave_5/RestoreV2:47*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
Ŕ
save_5/Assign_48Assignvc/dense_2/kernel/Adamsave_5/RestoreV2:48*
validate_shape(*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(
Â
save_5/Assign_49Assignvc/dense_2/kernel/Adam_1save_5/RestoreV2:49*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	*
use_locking(
Ż
save_5/Assign_50Assignvf/dense/biassave_5/RestoreV2:50*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0*
use_locking(
´
save_5/Assign_51Assignvf/dense/bias/Adamsave_5/RestoreV2:51*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(* 
_class
loc:@vf/dense/bias
ś
save_5/Assign_52Assignvf/dense/bias/Adam_1save_5/RestoreV2:52*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(*
T0*
_output_shapes	
:
ˇ
save_5/Assign_53Assignvf/dense/kernelsave_5/RestoreV2:53*
_output_shapes
:	<*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(
ź
save_5/Assign_54Assignvf/dense/kernel/Adamsave_5/RestoreV2:54*
T0*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@vf/dense/kernel*
use_locking(
ž
save_5/Assign_55Assignvf/dense/kernel/Adam_1save_5/RestoreV2:55*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
use_locking(*
validate_shape(
ł
save_5/Assign_56Assignvf/dense_1/biassave_5/RestoreV2:56*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(*"
_class
loc:@vf/dense_1/bias
¸
save_5/Assign_57Assignvf/dense_1/bias/Adamsave_5/RestoreV2:57*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(*
T0
ş
save_5/Assign_58Assignvf/dense_1/bias/Adam_1save_5/RestoreV2:58*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias
ź
save_5/Assign_59Assignvf/dense_1/kernelsave_5/RestoreV2:59*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:

Á
save_5/Assign_60Assignvf/dense_1/kernel/Adamsave_5/RestoreV2:60*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0
Ă
save_5/Assign_61Assignvf/dense_1/kernel/Adam_1save_5/RestoreV2:61*
validate_shape(*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(
˛
save_5/Assign_62Assignvf/dense_2/biassave_5/RestoreV2:62*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
use_locking(*
validate_shape(
ˇ
save_5/Assign_63Assignvf/dense_2/bias/Adamsave_5/RestoreV2:63*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:
š
save_5/Assign_64Assignvf/dense_2/bias/Adam_1save_5/RestoreV2:64*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
ť
save_5/Assign_65Assignvf/dense_2/kernelsave_5/RestoreV2:65*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0*
use_locking(
Ŕ
save_5/Assign_66Assignvf/dense_2/kernel/Adamsave_5/RestoreV2:66*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0
Â
save_5/Assign_67Assignvf/dense_2/kernel/Adam_1save_5/RestoreV2:67*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel


save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_2^save_5/Assign_20^save_5/Assign_21^save_5/Assign_22^save_5/Assign_23^save_5/Assign_24^save_5/Assign_25^save_5/Assign_26^save_5/Assign_27^save_5/Assign_28^save_5/Assign_29^save_5/Assign_3^save_5/Assign_30^save_5/Assign_31^save_5/Assign_32^save_5/Assign_33^save_5/Assign_34^save_5/Assign_35^save_5/Assign_36^save_5/Assign_37^save_5/Assign_38^save_5/Assign_39^save_5/Assign_4^save_5/Assign_40^save_5/Assign_41^save_5/Assign_42^save_5/Assign_43^save_5/Assign_44^save_5/Assign_45^save_5/Assign_46^save_5/Assign_47^save_5/Assign_48^save_5/Assign_49^save_5/Assign_5^save_5/Assign_50^save_5/Assign_51^save_5/Assign_52^save_5/Assign_53^save_5/Assign_54^save_5/Assign_55^save_5/Assign_56^save_5/Assign_57^save_5/Assign_58^save_5/Assign_59^save_5/Assign_6^save_5/Assign_60^save_5/Assign_61^save_5/Assign_62^save_5/Assign_63^save_5/Assign_64^save_5/Assign_65^save_5/Assign_66^save_5/Assign_67^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard
[
save_6/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
_output_shapes
: *
shape: *
dtype0

save_6/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_f59df07db083498cb73ae1eafda92894/part
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_6/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_6/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
÷
save_6/SaveV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
đ
save_6/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ň
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*)
_class
loc:@save_6/ShardedFilename*
T0*
_output_shapes
: 
Ł
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
T0*
_output_shapes
:*

axis *
N

save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(

save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
_output_shapes
: *
T0
ú
save_6/RestoreV2/tensor_namesConst*
_output_shapes
:D*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ó
!save_6/RestoreV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ę
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ş
save_6/AssignAssignbeta1_powersave_6/RestoreV2*
validate_shape(*
T0*
_output_shapes
: *
use_locking(*(
_class
loc:@penalty/penalty_param
°
save_6/Assign_1Assignbeta1_power_1save_6/RestoreV2:1*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: *
T0*
use_locking(
¨
save_6/Assign_2Assignbeta1_power_2save_6/RestoreV2:2*
T0*
_output_shapes
: *
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(
Ž
save_6/Assign_3Assignbeta2_powersave_6/RestoreV2:3*
validate_shape(*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
°
save_6/Assign_4Assignbeta2_power_1save_6/RestoreV2:4*
_output_shapes
: *
T0*
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(
¨
save_6/Assign_5Assignbeta2_power_2save_6/RestoreV2:5*
use_locking(*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias*
validate_shape(
¸
save_6/Assign_6Assignpenalty/penalty_paramsave_6/RestoreV2:6*
use_locking(*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0
˝
save_6/Assign_7Assignpenalty/penalty_param/Adamsave_6/RestoreV2:7*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param*
use_locking(*
T0
ż
save_6/Assign_8Assignpenalty/penalty_param/Adam_1save_6/RestoreV2:8*
use_locking(*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(
ż
save_6/Assign_9Assignpenalty/penalty_param/Adam_2save_6/RestoreV2:9*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0*
use_locking(
Á
save_6/Assign_10Assignpenalty/penalty_param/Adam_3save_6/RestoreV2:10*
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0*
use_locking(*
_output_shapes
: 
Ż
save_6/Assign_11Assignpi/dense/biassave_6/RestoreV2:11*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
_output_shapes	
:
´
save_6/Assign_12Assignpi/dense/bias/Adamsave_6/RestoreV2:12*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:
ś
save_6/Assign_13Assignpi/dense/bias/Adam_1save_6/RestoreV2:13*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
ˇ
save_6/Assign_14Assignpi/dense/kernelsave_6/RestoreV2:14*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	<*
T0
ź
save_6/Assign_15Assignpi/dense/kernel/Adamsave_6/RestoreV2:15*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
use_locking(*
validate_shape(*
T0
ž
save_6/Assign_16Assignpi/dense/kernel/Adam_1save_6/RestoreV2:16*
T0*
use_locking(*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
validate_shape(
ł
save_6/Assign_17Assignpi/dense_1/biassave_6/RestoreV2:17*
validate_shape(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0
¸
save_6/Assign_18Assignpi/dense_1/bias/Adamsave_6/RestoreV2:18*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias
ş
save_6/Assign_19Assignpi/dense_1/bias/Adam_1save_6/RestoreV2:19*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0
ź
save_6/Assign_20Assignpi/dense_1/kernelsave_6/RestoreV2:20*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel
Á
save_6/Assign_21Assignpi/dense_1/kernel/Adamsave_6/RestoreV2:21* 
_output_shapes
:
*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0
Ă
save_6/Assign_22Assignpi/dense_1/kernel/Adam_1save_6/RestoreV2:22*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:

˛
save_6/Assign_23Assignpi/dense_2/biassave_6/RestoreV2:23*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(
ˇ
save_6/Assign_24Assignpi/dense_2/bias/Adamsave_6/RestoreV2:24*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
š
save_6/Assign_25Assignpi/dense_2/bias/Adam_1save_6/RestoreV2:25*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
ť
save_6/Assign_26Assignpi/dense_2/kernelsave_6/RestoreV2:26*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
:	*
validate_shape(*
use_locking(
Ŕ
save_6/Assign_27Assignpi/dense_2/kernel/Adamsave_6/RestoreV2:27*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	
Â
save_6/Assign_28Assignpi/dense_2/kernel/Adam_1save_6/RestoreV2:28*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
validate_shape(
¨
save_6/Assign_29Assign
pi/log_stdsave_6/RestoreV2:29*
_output_shapes
:*
T0*
use_locking(*
_class
loc:@pi/log_std*
validate_shape(
­
save_6/Assign_30Assignpi/log_std/Adamsave_6/RestoreV2:30*
_class
loc:@pi/log_std*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
Ż
save_6/Assign_31Assignpi/log_std/Adam_1save_6/RestoreV2:31*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*
_class
loc:@pi/log_std
Ż
save_6/Assign_32Assignvc/dense/biassave_6/RestoreV2:32*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias
´
save_6/Assign_33Assignvc/dense/bias/Adamsave_6/RestoreV2:33*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0
ś
save_6/Assign_34Assignvc/dense/bias/Adam_1save_6/RestoreV2:34*
T0* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes	
:*
validate_shape(
ˇ
save_6/Assign_35Assignvc/dense/kernelsave_6/RestoreV2:35*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<
ź
save_6/Assign_36Assignvc/dense/kernel/Adamsave_6/RestoreV2:36*
use_locking(*
validate_shape(*
_output_shapes
:	<*
T0*"
_class
loc:@vc/dense/kernel
ž
save_6/Assign_37Assignvc/dense/kernel/Adam_1save_6/RestoreV2:37*
T0*
_output_shapes
:	<*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel
ł
save_6/Assign_38Assignvc/dense_1/biassave_6/RestoreV2:38*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0
¸
save_6/Assign_39Assignvc/dense_1/bias/Adamsave_6/RestoreV2:39*
T0*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
use_locking(
ş
save_6/Assign_40Assignvc/dense_1/bias/Adam_1save_6/RestoreV2:40*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:
ź
save_6/Assign_41Assignvc/dense_1/kernelsave_6/RestoreV2:41*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:

Á
save_6/Assign_42Assignvc/dense_1/kernel/Adamsave_6/RestoreV2:42* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(*
use_locking(
Ă
save_6/Assign_43Assignvc/dense_1/kernel/Adam_1save_6/RestoreV2:43*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:
*
use_locking(
˛
save_6/Assign_44Assignvc/dense_2/biassave_6/RestoreV2:44*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(
ˇ
save_6/Assign_45Assignvc/dense_2/bias/Adamsave_6/RestoreV2:45*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
š
save_6/Assign_46Assignvc/dense_2/bias/Adam_1save_6/RestoreV2:46*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0
ť
save_6/Assign_47Assignvc/dense_2/kernelsave_6/RestoreV2:47*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	
Ŕ
save_6/Assign_48Assignvc/dense_2/kernel/Adamsave_6/RestoreV2:48*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	*
use_locking(
Â
save_6/Assign_49Assignvc/dense_2/kernel/Adam_1save_6/RestoreV2:49*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
Ż
save_6/Assign_50Assignvf/dense/biassave_6/RestoreV2:50*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
T0*
use_locking(
´
save_6/Assign_51Assignvf/dense/bias/Adamsave_6/RestoreV2:51* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
ś
save_6/Assign_52Assignvf/dense/bias/Adam_1save_6/RestoreV2:52*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0* 
_class
loc:@vf/dense/bias
ˇ
save_6/Assign_53Assignvf/dense/kernelsave_6/RestoreV2:53*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	<*
validate_shape(
ź
save_6/Assign_54Assignvf/dense/kernel/Adamsave_6/RestoreV2:54*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	<*
use_locking(
ž
save_6/Assign_55Assignvf/dense/kernel/Adam_1save_6/RestoreV2:55*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<
ł
save_6/Assign_56Assignvf/dense_1/biassave_6/RestoreV2:56*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0
¸
save_6/Assign_57Assignvf/dense_1/bias/Adamsave_6/RestoreV2:57*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:
ş
save_6/Assign_58Assignvf/dense_1/bias/Adam_1save_6/RestoreV2:58*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias
ź
save_6/Assign_59Assignvf/dense_1/kernelsave_6/RestoreV2:59*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(
Á
save_6/Assign_60Assignvf/dense_1/kernel/Adamsave_6/RestoreV2:60*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0
Ă
save_6/Assign_61Assignvf/dense_1/kernel/Adam_1save_6/RestoreV2:61*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0
˛
save_6/Assign_62Assignvf/dense_2/biassave_6/RestoreV2:62*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
ˇ
save_6/Assign_63Assignvf/dense_2/bias/Adamsave_6/RestoreV2:63*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias
š
save_6/Assign_64Assignvf/dense_2/bias/Adam_1save_6/RestoreV2:64*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_2/bias
ť
save_6/Assign_65Assignvf/dense_2/kernelsave_6/RestoreV2:65*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(
Ŕ
save_6/Assign_66Assignvf/dense_2/kernel/Adamsave_6/RestoreV2:66*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(
Â
save_6/Assign_67Assignvf/dense_2/kernel/Adam_1save_6/RestoreV2:67*
_output_shapes
:	*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0*
validate_shape(


save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_2^save_6/Assign_20^save_6/Assign_21^save_6/Assign_22^save_6/Assign_23^save_6/Assign_24^save_6/Assign_25^save_6/Assign_26^save_6/Assign_27^save_6/Assign_28^save_6/Assign_29^save_6/Assign_3^save_6/Assign_30^save_6/Assign_31^save_6/Assign_32^save_6/Assign_33^save_6/Assign_34^save_6/Assign_35^save_6/Assign_36^save_6/Assign_37^save_6/Assign_38^save_6/Assign_39^save_6/Assign_4^save_6/Assign_40^save_6/Assign_41^save_6/Assign_42^save_6/Assign_43^save_6/Assign_44^save_6/Assign_45^save_6/Assign_46^save_6/Assign_47^save_6/Assign_48^save_6/Assign_49^save_6/Assign_5^save_6/Assign_50^save_6/Assign_51^save_6/Assign_52^save_6/Assign_53^save_6/Assign_54^save_6/Assign_55^save_6/Assign_56^save_6/Assign_57^save_6/Assign_58^save_6/Assign_59^save_6/Assign_6^save_6/Assign_60^save_6/Assign_61^save_6/Assign_62^save_6/Assign_63^save_6/Assign_64^save_6/Assign_65^save_6/Assign_66^save_6/Assign_67^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard
[
save_7/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_7/filenamePlaceholderWithDefaultsave_7/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_7/ConstPlaceholderWithDefaultsave_7/filename*
dtype0*
shape: *
_output_shapes
: 

save_7/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_5c684f38c6f14f9faf1ba0ef04988ae4/part*
dtype0
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
S
save_7/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_7/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
÷
save_7/SaveV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
đ
save_7/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ň
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_7/ShardedFilename
Ł
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*

axis *
N*
T0*
_output_shapes
:

save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(

save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency*
T0*
_output_shapes
: 
ú
save_7/RestoreV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D*
dtype0
ó
!save_7/RestoreV2/shape_and_slicesConst*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ę
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ş
save_7/AssignAssignbeta1_powersave_7/RestoreV2*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
°
save_7/Assign_1Assignbeta1_power_1save_7/RestoreV2:1*
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(
¨
save_7/Assign_2Assignbeta1_power_2save_7/RestoreV2:2*
validate_shape(*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias*
use_locking(
Ž
save_7/Assign_3Assignbeta2_powersave_7/RestoreV2:3*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(
°
save_7/Assign_4Assignbeta2_power_1save_7/RestoreV2:4*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
¨
save_7/Assign_5Assignbeta2_power_2save_7/RestoreV2:5* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
¸
save_7/Assign_6Assignpenalty/penalty_paramsave_7/RestoreV2:6*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
T0
˝
save_7/Assign_7Assignpenalty/penalty_param/Adamsave_7/RestoreV2:7*
use_locking(*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
ż
save_7/Assign_8Assignpenalty/penalty_param/Adam_1save_7/RestoreV2:8*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
ż
save_7/Assign_9Assignpenalty/penalty_param/Adam_2save_7/RestoreV2:9*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
Á
save_7/Assign_10Assignpenalty/penalty_param/Adam_3save_7/RestoreV2:10*
T0*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: *
validate_shape(
Ż
save_7/Assign_11Assignpi/dense/biassave_7/RestoreV2:11*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
T0
´
save_7/Assign_12Assignpi/dense/bias/Adamsave_7/RestoreV2:12*
_output_shapes	
:*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(
ś
save_7/Assign_13Assignpi/dense/bias/Adam_1save_7/RestoreV2:13*
_output_shapes	
:*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(
ˇ
save_7/Assign_14Assignpi/dense/kernelsave_7/RestoreV2:14*
validate_shape(*
use_locking(*
_output_shapes
:	<*
T0*"
_class
loc:@pi/dense/kernel
ź
save_7/Assign_15Assignpi/dense/kernel/Adamsave_7/RestoreV2:15*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<
ž
save_7/Assign_16Assignpi/dense/kernel/Adam_1save_7/RestoreV2:16*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<
ł
save_7/Assign_17Assignpi/dense_1/biassave_7/RestoreV2:17*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(
¸
save_7/Assign_18Assignpi/dense_1/bias/Adamsave_7/RestoreV2:18*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
validate_shape(
ş
save_7/Assign_19Assignpi/dense_1/bias/Adam_1save_7/RestoreV2:19*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias
ź
save_7/Assign_20Assignpi/dense_1/kernelsave_7/RestoreV2:20*
T0*
validate_shape(*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel
Á
save_7/Assign_21Assignpi/dense_1/kernel/Adamsave_7/RestoreV2:21*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(
Ă
save_7/Assign_22Assignpi/dense_1/kernel/Adam_1save_7/RestoreV2:22*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:

˛
save_7/Assign_23Assignpi/dense_2/biassave_7/RestoreV2:23*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(*
_output_shapes
:
ˇ
save_7/Assign_24Assignpi/dense_2/bias/Adamsave_7/RestoreV2:24*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias
š
save_7/Assign_25Assignpi/dense_2/bias/Adam_1save_7/RestoreV2:25*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
ť
save_7/Assign_26Assignpi/dense_2/kernelsave_7/RestoreV2:26*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
Ŕ
save_7/Assign_27Assignpi/dense_2/kernel/Adamsave_7/RestoreV2:27*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
use_locking(
Â
save_7/Assign_28Assignpi/dense_2/kernel/Adam_1save_7/RestoreV2:28*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
validate_shape(
¨
save_7/Assign_29Assign
pi/log_stdsave_7/RestoreV2:29*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
­
save_7/Assign_30Assignpi/log_std/Adamsave_7/RestoreV2:30*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
Ż
save_7/Assign_31Assignpi/log_std/Adam_1save_7/RestoreV2:31*
_class
loc:@pi/log_std*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
Ż
save_7/Assign_32Assignvc/dense/biassave_7/RestoreV2:32*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0
´
save_7/Assign_33Assignvc/dense/bias/Adamsave_7/RestoreV2:33*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
validate_shape(
ś
save_7/Assign_34Assignvc/dense/bias/Adam_1save_7/RestoreV2:34*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(* 
_class
loc:@vc/dense/bias
ˇ
save_7/Assign_35Assignvc/dense/kernelsave_7/RestoreV2:35*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	<
ź
save_7/Assign_36Assignvc/dense/kernel/Adamsave_7/RestoreV2:36*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
T0*
use_locking(*
validate_shape(
ž
save_7/Assign_37Assignvc/dense/kernel/Adam_1save_7/RestoreV2:37*
validate_shape(*
T0*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vc/dense/kernel
ł
save_7/Assign_38Assignvc/dense_1/biassave_7/RestoreV2:38*"
_class
loc:@vc/dense_1/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:
¸
save_7/Assign_39Assignvc/dense_1/bias/Adamsave_7/RestoreV2:39*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias
ş
save_7/Assign_40Assignvc/dense_1/bias/Adam_1save_7/RestoreV2:40*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
ź
save_7/Assign_41Assignvc/dense_1/kernelsave_7/RestoreV2:41* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(*
T0
Á
save_7/Assign_42Assignvc/dense_1/kernel/Adamsave_7/RestoreV2:42*
validate_shape(*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0
Ă
save_7/Assign_43Assignvc/dense_1/kernel/Adam_1save_7/RestoreV2:43*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:

˛
save_7/Assign_44Assignvc/dense_2/biassave_7/RestoreV2:44*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
ˇ
save_7/Assign_45Assignvc/dense_2/bias/Adamsave_7/RestoreV2:45*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0*
_output_shapes
:
š
save_7/Assign_46Assignvc/dense_2/bias/Adam_1save_7/RestoreV2:46*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(
ť
save_7/Assign_47Assignvc/dense_2/kernelsave_7/RestoreV2:47*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
Ŕ
save_7/Assign_48Assignvc/dense_2/kernel/Adamsave_7/RestoreV2:48*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@vc/dense_2/kernel
Â
save_7/Assign_49Assignvc/dense_2/kernel/Adam_1save_7/RestoreV2:49*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
T0*
use_locking(
Ż
save_7/Assign_50Assignvf/dense/biassave_7/RestoreV2:50*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
´
save_7/Assign_51Assignvf/dense/bias/Adamsave_7/RestoreV2:51*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vf/dense/bias*
use_locking(
ś
save_7/Assign_52Assignvf/dense/bias/Adam_1save_7/RestoreV2:52*
use_locking(*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
validate_shape(
ˇ
save_7/Assign_53Assignvf/dense/kernelsave_7/RestoreV2:53*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	<*
T0
ź
save_7/Assign_54Assignvf/dense/kernel/Adamsave_7/RestoreV2:54*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	<*
validate_shape(*
use_locking(
ž
save_7/Assign_55Assignvf/dense/kernel/Adam_1save_7/RestoreV2:55*
validate_shape(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
use_locking(*
T0
ł
save_7/Assign_56Assignvf/dense_1/biassave_7/RestoreV2:56*
validate_shape(*
_output_shapes	
:*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0
¸
save_7/Assign_57Assignvf/dense_1/bias/Adamsave_7/RestoreV2:57*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0
ş
save_7/Assign_58Assignvf/dense_1/bias/Adam_1save_7/RestoreV2:58*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_1/bias
ź
save_7/Assign_59Assignvf/dense_1/kernelsave_7/RestoreV2:59*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
T0*
use_locking(
Á
save_7/Assign_60Assignvf/dense_1/kernel/Adamsave_7/RestoreV2:60*
validate_shape(*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(
Ă
save_7/Assign_61Assignvf/dense_1/kernel/Adam_1save_7/RestoreV2:61*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:

˛
save_7/Assign_62Assignvf/dense_2/biassave_7/RestoreV2:62*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
use_locking(
ˇ
save_7/Assign_63Assignvf/dense_2/bias/Adamsave_7/RestoreV2:63*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(
š
save_7/Assign_64Assignvf/dense_2/bias/Adam_1save_7/RestoreV2:64*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
ť
save_7/Assign_65Assignvf/dense_2/kernelsave_7/RestoreV2:65*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
validate_shape(
Ŕ
save_7/Assign_66Assignvf/dense_2/kernel/Adamsave_7/RestoreV2:66*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(
Â
save_7/Assign_67Assignvf/dense_2/kernel/Adam_1save_7/RestoreV2:67*$
_class
loc:@vf/dense_2/kernel*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(


save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_2^save_7/Assign_20^save_7/Assign_21^save_7/Assign_22^save_7/Assign_23^save_7/Assign_24^save_7/Assign_25^save_7/Assign_26^save_7/Assign_27^save_7/Assign_28^save_7/Assign_29^save_7/Assign_3^save_7/Assign_30^save_7/Assign_31^save_7/Assign_32^save_7/Assign_33^save_7/Assign_34^save_7/Assign_35^save_7/Assign_36^save_7/Assign_37^save_7/Assign_38^save_7/Assign_39^save_7/Assign_4^save_7/Assign_40^save_7/Assign_41^save_7/Assign_42^save_7/Assign_43^save_7/Assign_44^save_7/Assign_45^save_7/Assign_46^save_7/Assign_47^save_7/Assign_48^save_7/Assign_49^save_7/Assign_5^save_7/Assign_50^save_7/Assign_51^save_7/Assign_52^save_7/Assign_53^save_7/Assign_54^save_7/Assign_55^save_7/Assign_56^save_7/Assign_57^save_7/Assign_58^save_7/Assign_59^save_7/Assign_6^save_7/Assign_60^save_7/Assign_61^save_7/Assign_62^save_7/Assign_63^save_7/Assign_64^save_7/Assign_65^save_7/Assign_66^save_7/Assign_67^save_7/Assign_7^save_7/Assign_8^save_7/Assign_9
1
save_7/restore_allNoOp^save_7/restore_shard
[
save_8/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_8/filenamePlaceholderWithDefaultsave_8/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_8/ConstPlaceholderWithDefaultsave_8/filename*
dtype0*
shape: *
_output_shapes
: 

save_8/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_3f810614ba75493b922439158842dc53/part*
dtype0
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_8/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_8/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
÷
save_8/SaveV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D*
dtype0
đ
save_8/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ň
save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_8/ShardedFilename
Ł
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilename^save_8/control_dependency*

axis *
N*
_output_shapes
:*
T0

save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(

save_8/IdentityIdentitysave_8/Const^save_8/MergeV2Checkpoints^save_8/control_dependency*
T0*
_output_shapes
: 
ú
save_8/RestoreV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
ó
!save_8/RestoreV2/shape_and_slicesConst*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ę
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ş
save_8/AssignAssignbeta1_powersave_8/RestoreV2*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
T0
°
save_8/Assign_1Assignbeta1_power_1save_8/RestoreV2:1*
T0*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
¨
save_8/Assign_2Assignbeta1_power_2save_8/RestoreV2:2*
validate_shape(*
T0*
use_locking(*
_output_shapes
: * 
_class
loc:@vc/dense/bias
Ž
save_8/Assign_3Assignbeta2_powersave_8/RestoreV2:3*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
°
save_8/Assign_4Assignbeta2_power_1save_8/RestoreV2:4*
validate_shape(*
use_locking(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
¨
save_8/Assign_5Assignbeta2_power_2save_8/RestoreV2:5* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes
: *
T0*
validate_shape(
¸
save_8/Assign_6Assignpenalty/penalty_paramsave_8/RestoreV2:6*
_output_shapes
: *
validate_shape(*
T0*
use_locking(*(
_class
loc:@penalty/penalty_param
˝
save_8/Assign_7Assignpenalty/penalty_param/Adamsave_8/RestoreV2:7*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param
ż
save_8/Assign_8Assignpenalty/penalty_param/Adam_1save_8/RestoreV2:8*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0*
use_locking(
ż
save_8/Assign_9Assignpenalty/penalty_param/Adam_2save_8/RestoreV2:9*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0*
use_locking(*
validate_shape(
Á
save_8/Assign_10Assignpenalty/penalty_param/Adam_3save_8/RestoreV2:10*
validate_shape(*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0
Ż
save_8/Assign_11Assignpi/dense/biassave_8/RestoreV2:11*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias
´
save_8/Assign_12Assignpi/dense/bias/Adamsave_8/RestoreV2:12*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes	
:
ś
save_8/Assign_13Assignpi/dense/bias/Adam_1save_8/RestoreV2:13*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes	
:*
validate_shape(
ˇ
save_8/Assign_14Assignpi/dense/kernelsave_8/RestoreV2:14*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	<
ź
save_8/Assign_15Assignpi/dense/kernel/Adamsave_8/RestoreV2:15*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0
ž
save_8/Assign_16Assignpi/dense/kernel/Adam_1save_8/RestoreV2:16*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<
ł
save_8/Assign_17Assignpi/dense_1/biassave_8/RestoreV2:17*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:
¸
save_8/Assign_18Assignpi/dense_1/bias/Adamsave_8/RestoreV2:18*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:
ş
save_8/Assign_19Assignpi/dense_1/bias/Adam_1save_8/RestoreV2:19*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(
ź
save_8/Assign_20Assignpi/dense_1/kernelsave_8/RestoreV2:20*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
T0
Á
save_8/Assign_21Assignpi/dense_1/kernel/Adamsave_8/RestoreV2:21*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:
*
use_locking(
Ă
save_8/Assign_22Assignpi/dense_1/kernel/Adam_1save_8/RestoreV2:22*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:
*
use_locking(
˛
save_8/Assign_23Assignpi/dense_2/biassave_8/RestoreV2:23*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
use_locking(
ˇ
save_8/Assign_24Assignpi/dense_2/bias/Adamsave_8/RestoreV2:24*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
š
save_8/Assign_25Assignpi/dense_2/bias/Adam_1save_8/RestoreV2:25*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(
ť
save_8/Assign_26Assignpi/dense_2/kernelsave_8/RestoreV2:26*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	
Ŕ
save_8/Assign_27Assignpi/dense_2/kernel/Adamsave_8/RestoreV2:27*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes
:	
Â
save_8/Assign_28Assignpi/dense_2/kernel/Adam_1save_8/RestoreV2:28*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(
¨
save_8/Assign_29Assign
pi/log_stdsave_8/RestoreV2:29*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
­
save_8/Assign_30Assignpi/log_std/Adamsave_8/RestoreV2:30*
T0*
validate_shape(*
_class
loc:@pi/log_std*
_output_shapes
:*
use_locking(
Ż
save_8/Assign_31Assignpi/log_std/Adam_1save_8/RestoreV2:31*
validate_shape(*
_output_shapes
:*
_class
loc:@pi/log_std*
T0*
use_locking(
Ż
save_8/Assign_32Assignvc/dense/biassave_8/RestoreV2:32*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(* 
_class
loc:@vc/dense/bias
´
save_8/Assign_33Assignvc/dense/bias/Adamsave_8/RestoreV2:33*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
validate_shape(
ś
save_8/Assign_34Assignvc/dense/bias/Adam_1save_8/RestoreV2:34*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0* 
_class
loc:@vc/dense/bias
ˇ
save_8/Assign_35Assignvc/dense/kernelsave_8/RestoreV2:35*
use_locking(*
T0*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@vc/dense/kernel
ź
save_8/Assign_36Assignvc/dense/kernel/Adamsave_8/RestoreV2:36*
use_locking(*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
T0
ž
save_8/Assign_37Assignvc/dense/kernel/Adam_1save_8/RestoreV2:37*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
use_locking(
ł
save_8/Assign_38Assignvc/dense_1/biassave_8/RestoreV2:38*
T0*
use_locking(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(
¸
save_8/Assign_39Assignvc/dense_1/bias/Adamsave_8/RestoreV2:39*
use_locking(*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
T0
ş
save_8/Assign_40Assignvc/dense_1/bias/Adam_1save_8/RestoreV2:40*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
ź
save_8/Assign_41Assignvc/dense_1/kernelsave_8/RestoreV2:41*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:

Á
save_8/Assign_42Assignvc/dense_1/kernel/Adamsave_8/RestoreV2:42*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0
Ă
save_8/Assign_43Assignvc/dense_1/kernel/Adam_1save_8/RestoreV2:43*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
validate_shape(
˛
save_8/Assign_44Assignvc/dense_2/biassave_8/RestoreV2:44*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias
ˇ
save_8/Assign_45Assignvc/dense_2/bias/Adamsave_8/RestoreV2:45*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(*
T0
š
save_8/Assign_46Assignvc/dense_2/bias/Adam_1save_8/RestoreV2:46*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias
ť
save_8/Assign_47Assignvc/dense_2/kernelsave_8/RestoreV2:47*$
_class
loc:@vc/dense_2/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	
Ŕ
save_8/Assign_48Assignvc/dense_2/kernel/Adamsave_8/RestoreV2:48*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_8/Assign_49Assignvc/dense_2/kernel/Adam_1save_8/RestoreV2:49*
_output_shapes
:	*
T0*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(
Ż
save_8/Assign_50Assignvf/dense/biassave_8/RestoreV2:50*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
use_locking(
´
save_8/Assign_51Assignvf/dense/bias/Adamsave_8/RestoreV2:51* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
ś
save_8/Assign_52Assignvf/dense/bias/Adam_1save_8/RestoreV2:52*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vf/dense/bias
ˇ
save_8/Assign_53Assignvf/dense/kernelsave_8/RestoreV2:53*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(*
T0
ź
save_8/Assign_54Assignvf/dense/kernel/Adamsave_8/RestoreV2:54*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel
ž
save_8/Assign_55Assignvf/dense/kernel/Adam_1save_8/RestoreV2:55*
_output_shapes
:	<*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(
ł
save_8/Assign_56Assignvf/dense_1/biassave_8/RestoreV2:56*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0
¸
save_8/Assign_57Assignvf/dense_1/bias/Adamsave_8/RestoreV2:57*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(
ş
save_8/Assign_58Assignvf/dense_1/bias/Adam_1save_8/RestoreV2:58*
T0*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(
ź
save_8/Assign_59Assignvf/dense_1/kernelsave_8/RestoreV2:59* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
T0*
use_locking(
Á
save_8/Assign_60Assignvf/dense_1/kernel/Adamsave_8/RestoreV2:60*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(
Ă
save_8/Assign_61Assignvf/dense_1/kernel/Adam_1save_8/RestoreV2:61*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(
˛
save_8/Assign_62Assignvf/dense_2/biassave_8/RestoreV2:62*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
ˇ
save_8/Assign_63Assignvf/dense_2/bias/Adamsave_8/RestoreV2:63*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
use_locking(*
T0*
validate_shape(
š
save_8/Assign_64Assignvf/dense_2/bias/Adam_1save_8/RestoreV2:64*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
ť
save_8/Assign_65Assignvf/dense_2/kernelsave_8/RestoreV2:65*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0
Ŕ
save_8/Assign_66Assignvf/dense_2/kernel/Adamsave_8/RestoreV2:66*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(
Â
save_8/Assign_67Assignvf/dense_2/kernel/Adam_1save_8/RestoreV2:67*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	*
T0


save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_16^save_8/Assign_17^save_8/Assign_18^save_8/Assign_19^save_8/Assign_2^save_8/Assign_20^save_8/Assign_21^save_8/Assign_22^save_8/Assign_23^save_8/Assign_24^save_8/Assign_25^save_8/Assign_26^save_8/Assign_27^save_8/Assign_28^save_8/Assign_29^save_8/Assign_3^save_8/Assign_30^save_8/Assign_31^save_8/Assign_32^save_8/Assign_33^save_8/Assign_34^save_8/Assign_35^save_8/Assign_36^save_8/Assign_37^save_8/Assign_38^save_8/Assign_39^save_8/Assign_4^save_8/Assign_40^save_8/Assign_41^save_8/Assign_42^save_8/Assign_43^save_8/Assign_44^save_8/Assign_45^save_8/Assign_46^save_8/Assign_47^save_8/Assign_48^save_8/Assign_49^save_8/Assign_5^save_8/Assign_50^save_8/Assign_51^save_8/Assign_52^save_8/Assign_53^save_8/Assign_54^save_8/Assign_55^save_8/Assign_56^save_8/Assign_57^save_8/Assign_58^save_8/Assign_59^save_8/Assign_6^save_8/Assign_60^save_8/Assign_61^save_8/Assign_62^save_8/Assign_63^save_8/Assign_64^save_8/Assign_65^save_8/Assign_66^save_8/Assign_67^save_8/Assign_7^save_8/Assign_8^save_8/Assign_9
1
save_8/restore_allNoOp^save_8/restore_shard
[
save_9/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_9/filenamePlaceholderWithDefaultsave_9/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_9/ConstPlaceholderWithDefaultsave_9/filename*
shape: *
dtype0*
_output_shapes
: 

save_9/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_358ad789413d44d2a3d4e3f4fab99b4f/part
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
S
save_9/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_9/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
÷
save_9/SaveV2/tensor_namesConst*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:D
đ
save_9/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ň
save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_9/ShardedFilename
Ł
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilename^save_9/control_dependency*
T0*
N*

axis *
_output_shapes
:

save_9/MergeV2CheckpointsMergeV2Checkpoints-save_9/MergeV2Checkpoints/checkpoint_prefixessave_9/Const*
delete_old_dirs(

save_9/IdentityIdentitysave_9/Const^save_9/MergeV2Checkpoints^save_9/control_dependency*
T0*
_output_shapes
: 
ú
save_9/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:D*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ó
!save_9/RestoreV2/shape_and_slicesConst*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:D*
dtype0
ę
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ş
save_9/AssignAssignbeta1_powersave_9/RestoreV2*
T0*(
_class
loc:@penalty/penalty_param*
validate_shape(*
use_locking(*
_output_shapes
: 
°
save_9/Assign_1Assignbeta1_power_1save_9/RestoreV2:1*
validate_shape(*
T0*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
use_locking(
¨
save_9/Assign_2Assignbeta1_power_2save_9/RestoreV2:2*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: 
Ž
save_9/Assign_3Assignbeta2_powersave_9/RestoreV2:3*(
_class
loc:@penalty/penalty_param*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
°
save_9/Assign_4Assignbeta2_power_1save_9/RestoreV2:4*
T0*
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(*
_output_shapes
: 
¨
save_9/Assign_5Assignbeta2_power_2save_9/RestoreV2:5*
T0*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes
: 
¸
save_9/Assign_6Assignpenalty/penalty_paramsave_9/RestoreV2:6*
use_locking(*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param*
T0
˝
save_9/Assign_7Assignpenalty/penalty_param/Adamsave_9/RestoreV2:7*
use_locking(*(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0*
_output_shapes
: 
ż
save_9/Assign_8Assignpenalty/penalty_param/Adam_1save_9/RestoreV2:8*
use_locking(*
validate_shape(*
T0*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
ż
save_9/Assign_9Assignpenalty/penalty_param/Adam_2save_9/RestoreV2:9*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
T0*
use_locking(
Á
save_9/Assign_10Assignpenalty/penalty_param/Adam_3save_9/RestoreV2:10*(
_class
loc:@penalty/penalty_param*
use_locking(*
_output_shapes
: *
validate_shape(*
T0
Ż
save_9/Assign_11Assignpi/dense/biassave_9/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(
´
save_9/Assign_12Assignpi/dense/bias/Adamsave_9/RestoreV2:12*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
ś
save_9/Assign_13Assignpi/dense/bias/Adam_1save_9/RestoreV2:13*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
ˇ
save_9/Assign_14Assignpi/dense/kernelsave_9/RestoreV2:14*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
use_locking(*
validate_shape(
ź
save_9/Assign_15Assignpi/dense/kernel/Adamsave_9/RestoreV2:15*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0
ž
save_9/Assign_16Assignpi/dense/kernel/Adam_1save_9/RestoreV2:16*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(*
T0
ł
save_9/Assign_17Assignpi/dense_1/biassave_9/RestoreV2:17*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias
¸
save_9/Assign_18Assignpi/dense_1/bias/Adamsave_9/RestoreV2:18*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(*"
_class
loc:@pi/dense_1/bias
ş
save_9/Assign_19Assignpi/dense_1/bias/Adam_1save_9/RestoreV2:19*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0
ź
save_9/Assign_20Assignpi/dense_1/kernelsave_9/RestoreV2:20*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:

Á
save_9/Assign_21Assignpi/dense_1/kernel/Adamsave_9/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_9/Assign_22Assignpi/dense_1/kernel/Adam_1save_9/RestoreV2:22*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

˛
save_9/Assign_23Assignpi/dense_2/biassave_9/RestoreV2:23*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(
ˇ
save_9/Assign_24Assignpi/dense_2/bias/Adamsave_9/RestoreV2:24*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
š
save_9/Assign_25Assignpi/dense_2/bias/Adam_1save_9/RestoreV2:25*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias
ť
save_9/Assign_26Assignpi/dense_2/kernelsave_9/RestoreV2:26*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0
Ŕ
save_9/Assign_27Assignpi/dense_2/kernel/Adamsave_9/RestoreV2:27*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
validate_shape(
Â
save_9/Assign_28Assignpi/dense_2/kernel/Adam_1save_9/RestoreV2:28*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(*
T0
¨
save_9/Assign_29Assign
pi/log_stdsave_9/RestoreV2:29*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*
_class
loc:@pi/log_std
­
save_9/Assign_30Assignpi/log_std/Adamsave_9/RestoreV2:30*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*
_class
loc:@pi/log_std
Ż
save_9/Assign_31Assignpi/log_std/Adam_1save_9/RestoreV2:31*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*
_class
loc:@pi/log_std
Ż
save_9/Assign_32Assignvc/dense/biassave_9/RestoreV2:32*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0
´
save_9/Assign_33Assignvc/dense/bias/Adamsave_9/RestoreV2:33*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
ś
save_9/Assign_34Assignvc/dense/bias/Adam_1save_9/RestoreV2:34* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
ˇ
save_9/Assign_35Assignvc/dense/kernelsave_9/RestoreV2:35*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0*
_output_shapes
:	<*
validate_shape(
ź
save_9/Assign_36Assignvc/dense/kernel/Adamsave_9/RestoreV2:36*
validate_shape(*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0*
_output_shapes
:	<
ž
save_9/Assign_37Assignvc/dense/kernel/Adam_1save_9/RestoreV2:37*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel
ł
save_9/Assign_38Assignvc/dense_1/biassave_9/RestoreV2:38*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias
¸
save_9/Assign_39Assignvc/dense_1/bias/Adamsave_9/RestoreV2:39*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:
ş
save_9/Assign_40Assignvc/dense_1/bias/Adam_1save_9/RestoreV2:40*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
ź
save_9/Assign_41Assignvc/dense_1/kernelsave_9/RestoreV2:41*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0
Á
save_9/Assign_42Assignvc/dense_1/kernel/Adamsave_9/RestoreV2:42*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:

Ă
save_9/Assign_43Assignvc/dense_1/kernel/Adam_1save_9/RestoreV2:43*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(
˛
save_9/Assign_44Assignvc/dense_2/biassave_9/RestoreV2:44*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:
ˇ
save_9/Assign_45Assignvc/dense_2/bias/Adamsave_9/RestoreV2:45*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0
š
save_9/Assign_46Assignvc/dense_2/bias/Adam_1save_9/RestoreV2:46*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(
ť
save_9/Assign_47Assignvc/dense_2/kernelsave_9/RestoreV2:47*
validate_shape(*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	
Ŕ
save_9/Assign_48Assignvc/dense_2/kernel/Adamsave_9/RestoreV2:48*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
use_locking(*
validate_shape(
Â
save_9/Assign_49Assignvc/dense_2/kernel/Adam_1save_9/RestoreV2:49*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	
Ż
save_9/Assign_50Assignvf/dense/biassave_9/RestoreV2:50* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
´
save_9/Assign_51Assignvf/dense/bias/Adamsave_9/RestoreV2:51* 
_class
loc:@vf/dense/bias*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(
ś
save_9/Assign_52Assignvf/dense/bias/Adam_1save_9/RestoreV2:52*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(* 
_class
loc:@vf/dense/bias
ˇ
save_9/Assign_53Assignvf/dense/kernelsave_9/RestoreV2:53*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	<
ź
save_9/Assign_54Assignvf/dense/kernel/Adamsave_9/RestoreV2:54*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
validate_shape(*
T0
ž
save_9/Assign_55Assignvf/dense/kernel/Adam_1save_9/RestoreV2:55*
T0*
_output_shapes
:	<*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense/kernel
ł
save_9/Assign_56Assignvf/dense_1/biassave_9/RestoreV2:56*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(
¸
save_9/Assign_57Assignvf/dense_1/bias/Adamsave_9/RestoreV2:57*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(
ş
save_9/Assign_58Assignvf/dense_1/bias/Adam_1save_9/RestoreV2:58*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0*
validate_shape(
ź
save_9/Assign_59Assignvf/dense_1/kernelsave_9/RestoreV2:59*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel
Á
save_9/Assign_60Assignvf/dense_1/kernel/Adamsave_9/RestoreV2:60*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
Ă
save_9/Assign_61Assignvf/dense_1/kernel/Adam_1save_9/RestoreV2:61*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
˛
save_9/Assign_62Assignvf/dense_2/biassave_9/RestoreV2:62*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(
ˇ
save_9/Assign_63Assignvf/dense_2/bias/Adamsave_9/RestoreV2:63*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias
š
save_9/Assign_64Assignvf/dense_2/bias/Adam_1save_9/RestoreV2:64*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense_2/bias
ť
save_9/Assign_65Assignvf/dense_2/kernelsave_9/RestoreV2:65*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0
Ŕ
save_9/Assign_66Assignvf/dense_2/kernel/Adamsave_9/RestoreV2:66*
_output_shapes
:	*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0
Â
save_9/Assign_67Assignvf/dense_2/kernel/Adam_1save_9/RestoreV2:67*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(


save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_16^save_9/Assign_17^save_9/Assign_18^save_9/Assign_19^save_9/Assign_2^save_9/Assign_20^save_9/Assign_21^save_9/Assign_22^save_9/Assign_23^save_9/Assign_24^save_9/Assign_25^save_9/Assign_26^save_9/Assign_27^save_9/Assign_28^save_9/Assign_29^save_9/Assign_3^save_9/Assign_30^save_9/Assign_31^save_9/Assign_32^save_9/Assign_33^save_9/Assign_34^save_9/Assign_35^save_9/Assign_36^save_9/Assign_37^save_9/Assign_38^save_9/Assign_39^save_9/Assign_4^save_9/Assign_40^save_9/Assign_41^save_9/Assign_42^save_9/Assign_43^save_9/Assign_44^save_9/Assign_45^save_9/Assign_46^save_9/Assign_47^save_9/Assign_48^save_9/Assign_49^save_9/Assign_5^save_9/Assign_50^save_9/Assign_51^save_9/Assign_52^save_9/Assign_53^save_9/Assign_54^save_9/Assign_55^save_9/Assign_56^save_9/Assign_57^save_9/Assign_58^save_9/Assign_59^save_9/Assign_6^save_9/Assign_60^save_9/Assign_61^save_9/Assign_62^save_9/Assign_63^save_9/Assign_64^save_9/Assign_65^save_9/Assign_66^save_9/Assign_67^save_9/Assign_7^save_9/Assign_8^save_9/Assign_9
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
shape: *
_output_shapes
: *
dtype0
k
save_10/ConstPlaceholderWithDefaultsave_10/filename*
_output_shapes
: *
dtype0*
shape: 

save_10/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_d75d89c1b98e412a9558893b5d4eb96d/part*
_output_shapes
: 
~
save_10/StringJoin
StringJoinsave_10/Constsave_10/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
T
save_10/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_10/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 

save_10/ShardedFilenameShardedFilenamesave_10/StringJoinsave_10/ShardedFilename/shardsave_10/num_shards*
_output_shapes
: 
ř
save_10/SaveV2/tensor_namesConst*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:D
ń
save_10/SaveV2/shape_and_slicesConst*
_output_shapes
:D*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ö
save_10/SaveV2SaveV2save_10/ShardedFilenamesave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2penalty/penalty_parampenalty/penalty_param/Adampenalty/penalty_param/Adam_1penalty/penalty_param/Adam_2penalty/penalty_param/Adam_3pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*R
dtypesH
F2D

save_10/control_dependencyIdentitysave_10/ShardedFilename^save_10/SaveV2**
_class 
loc:@save_10/ShardedFilename*
T0*
_output_shapes
: 
Ś
.save_10/MergeV2Checkpoints/checkpoint_prefixesPacksave_10/ShardedFilename^save_10/control_dependency*
_output_shapes
:*

axis *
N*
T0

save_10/MergeV2CheckpointsMergeV2Checkpoints.save_10/MergeV2Checkpoints/checkpoint_prefixessave_10/Const*
delete_old_dirs(

save_10/IdentityIdentitysave_10/Const^save_10/MergeV2Checkpoints^save_10/control_dependency*
_output_shapes
: *
T0
ű
save_10/RestoreV2/tensor_namesConst*
_output_shapes
:D*
dtype0*¨
valueBDBbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bpenalty/penalty_paramBpenalty/penalty_param/AdamBpenalty/penalty_param/Adam_1Bpenalty/penalty_param/Adam_2Bpenalty/penalty_param/Adam_3Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ô
"save_10/RestoreV2/shape_and_slicesConst*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
î
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices*Ś
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D
Ź
save_10/AssignAssignbeta1_powersave_10/RestoreV2*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
validate_shape(*
T0
˛
save_10/Assign_1Assignbeta1_power_1save_10/RestoreV2:1*
validate_shape(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: *
use_locking(*
T0
Ş
save_10/Assign_2Assignbeta1_power_2save_10/RestoreV2:2*
T0*
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
use_locking(
°
save_10/Assign_3Assignbeta2_powersave_10/RestoreV2:3*
_output_shapes
: *
validate_shape(*
T0*
use_locking(*(
_class
loc:@penalty/penalty_param
˛
save_10/Assign_4Assignbeta2_power_1save_10/RestoreV2:4*
_output_shapes
: *
T0*(
_class
loc:@penalty/penalty_param*
use_locking(*
validate_shape(
Ş
save_10/Assign_5Assignbeta2_power_2save_10/RestoreV2:5*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: 
ş
save_10/Assign_6Assignpenalty/penalty_paramsave_10/RestoreV2:6*
use_locking(*
T0*
_output_shapes
: *
validate_shape(*(
_class
loc:@penalty/penalty_param
ż
save_10/Assign_7Assignpenalty/penalty_param/Adamsave_10/RestoreV2:7*
T0*
use_locking(*
validate_shape(*
_output_shapes
: *(
_class
loc:@penalty/penalty_param
Á
save_10/Assign_8Assignpenalty/penalty_param/Adam_1save_10/RestoreV2:8*
validate_shape(*
T0*
use_locking(*(
_class
loc:@penalty/penalty_param*
_output_shapes
: 
Á
save_10/Assign_9Assignpenalty/penalty_param/Adam_2save_10/RestoreV2:9*(
_class
loc:@penalty/penalty_param*
validate_shape(*
T0*
_output_shapes
: *
use_locking(
Ă
save_10/Assign_10Assignpenalty/penalty_param/Adam_3save_10/RestoreV2:10*
validate_shape(*(
_class
loc:@penalty/penalty_param*
T0*
use_locking(*
_output_shapes
: 
ą
save_10/Assign_11Assignpi/dense/biassave_10/RestoreV2:11*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
_output_shapes	
:
ś
save_10/Assign_12Assignpi/dense/bias/Adamsave_10/RestoreV2:12* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
¸
save_10/Assign_13Assignpi/dense/bias/Adam_1save_10/RestoreV2:13*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
š
save_10/Assign_14Assignpi/dense/kernelsave_10/RestoreV2:14*
use_locking(*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0
ž
save_10/Assign_15Assignpi/dense/kernel/Adamsave_10/RestoreV2:15*
T0*
use_locking(*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
validate_shape(
Ŕ
save_10/Assign_16Assignpi/dense/kernel/Adam_1save_10/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
validate_shape(
ľ
save_10/Assign_17Assignpi/dense_1/biassave_10/RestoreV2:17*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(
ş
save_10/Assign_18Assignpi/dense_1/bias/Adamsave_10/RestoreV2:18*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(*
_output_shapes	
:
ź
save_10/Assign_19Assignpi/dense_1/bias/Adam_1save_10/RestoreV2:19*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0
ž
save_10/Assign_20Assignpi/dense_1/kernelsave_10/RestoreV2:20*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel
Ă
save_10/Assign_21Assignpi/dense_1/kernel/Adamsave_10/RestoreV2:21*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(
Ĺ
save_10/Assign_22Assignpi/dense_1/kernel/Adam_1save_10/RestoreV2:22*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:

´
save_10/Assign_23Assignpi/dense_2/biassave_10/RestoreV2:23*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(*
validate_shape(
š
save_10/Assign_24Assignpi/dense_2/bias/Adamsave_10/RestoreV2:24*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(
ť
save_10/Assign_25Assignpi/dense_2/bias/Adam_1save_10/RestoreV2:25*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
˝
save_10/Assign_26Assignpi/dense_2/kernelsave_10/RestoreV2:26*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(
Â
save_10/Assign_27Assignpi/dense_2/kernel/Adamsave_10/RestoreV2:27*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
Ä
save_10/Assign_28Assignpi/dense_2/kernel/Adam_1save_10/RestoreV2:28*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
:	
Ş
save_10/Assign_29Assign
pi/log_stdsave_10/RestoreV2:29*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*
_class
loc:@pi/log_std
Ż
save_10/Assign_30Assignpi/log_std/Adamsave_10/RestoreV2:30*
validate_shape(*
use_locking(*
_class
loc:@pi/log_std*
T0*
_output_shapes
:
ą
save_10/Assign_31Assignpi/log_std/Adam_1save_10/RestoreV2:31*
_output_shapes
:*
use_locking(*
_class
loc:@pi/log_std*
T0*
validate_shape(
ą
save_10/Assign_32Assignvc/dense/biassave_10/RestoreV2:32*
_output_shapes	
:*
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0
ś
save_10/Assign_33Assignvc/dense/bias/Adamsave_10/RestoreV2:33* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
¸
save_10/Assign_34Assignvc/dense/bias/Adam_1save_10/RestoreV2:34* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:
š
save_10/Assign_35Assignvc/dense/kernelsave_10/RestoreV2:35*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(*
_output_shapes
:	<
ž
save_10/Assign_36Assignvc/dense/kernel/Adamsave_10/RestoreV2:36*
validate_shape(*
T0*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
use_locking(
Ŕ
save_10/Assign_37Assignvc/dense/kernel/Adam_1save_10/RestoreV2:37*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	<
ľ
save_10/Assign_38Assignvc/dense_1/biassave_10/RestoreV2:38*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(
ş
save_10/Assign_39Assignvc/dense_1/bias/Adamsave_10/RestoreV2:39*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
ź
save_10/Assign_40Assignvc/dense_1/bias/Adam_1save_10/RestoreV2:40*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(
ž
save_10/Assign_41Assignvc/dense_1/kernelsave_10/RestoreV2:41*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0
Ă
save_10/Assign_42Assignvc/dense_1/kernel/Adamsave_10/RestoreV2:42*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:

Ĺ
save_10/Assign_43Assignvc/dense_1/kernel/Adam_1save_10/RestoreV2:43*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking(
´
save_10/Assign_44Assignvc/dense_2/biassave_10/RestoreV2:44*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias
š
save_10/Assign_45Assignvc/dense_2/bias/Adamsave_10/RestoreV2:45*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ť
save_10/Assign_46Assignvc/dense_2/bias/Adam_1save_10/RestoreV2:46*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
˝
save_10/Assign_47Assignvc/dense_2/kernelsave_10/RestoreV2:47*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
Â
save_10/Assign_48Assignvc/dense_2/kernel/Adamsave_10/RestoreV2:48*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
Ä
save_10/Assign_49Assignvc/dense_2/kernel/Adam_1save_10/RestoreV2:49*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
ą
save_10/Assign_50Assignvf/dense/biassave_10/RestoreV2:50*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ś
save_10/Assign_51Assignvf/dense/bias/Adamsave_10/RestoreV2:51*
use_locking(* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0*
_output_shapes	
:
¸
save_10/Assign_52Assignvf/dense/bias/Adam_1save_10/RestoreV2:52* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
š
save_10/Assign_53Assignvf/dense/kernelsave_10/RestoreV2:53*
use_locking(*
_output_shapes
:	<*
T0*
validate_shape(*"
_class
loc:@vf/dense/kernel
ž
save_10/Assign_54Assignvf/dense/kernel/Adamsave_10/RestoreV2:54*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
validate_shape(
Ŕ
save_10/Assign_55Assignvf/dense/kernel/Adam_1save_10/RestoreV2:55*
validate_shape(*
T0*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vf/dense/kernel
ľ
save_10/Assign_56Assignvf/dense_1/biassave_10/RestoreV2:56*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:
ş
save_10/Assign_57Assignvf/dense_1/bias/Adamsave_10/RestoreV2:57*
T0*
_output_shapes	
:*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(
ź
save_10/Assign_58Assignvf/dense_1/bias/Adam_1save_10/RestoreV2:58*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(
ž
save_10/Assign_59Assignvf/dense_1/kernelsave_10/RestoreV2:59* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(*
use_locking(
Ă
save_10/Assign_60Assignvf/dense_1/kernel/Adamsave_10/RestoreV2:60*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
T0*
use_locking(
Ĺ
save_10/Assign_61Assignvf/dense_1/kernel/Adam_1save_10/RestoreV2:61*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:

´
save_10/Assign_62Assignvf/dense_2/biassave_10/RestoreV2:62*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
use_locking(
š
save_10/Assign_63Assignvf/dense_2/bias/Adamsave_10/RestoreV2:63*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@vf/dense_2/bias
ť
save_10/Assign_64Assignvf/dense_2/bias/Adam_1save_10/RestoreV2:64*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias
˝
save_10/Assign_65Assignvf/dense_2/kernelsave_10/RestoreV2:65*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel
Â
save_10/Assign_66Assignvf/dense_2/kernel/Adamsave_10/RestoreV2:66*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(
Ä
save_10/Assign_67Assignvf/dense_2/kernel/Adam_1save_10/RestoreV2:67*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(
á

save_10/restore_shardNoOp^save_10/Assign^save_10/Assign_1^save_10/Assign_10^save_10/Assign_11^save_10/Assign_12^save_10/Assign_13^save_10/Assign_14^save_10/Assign_15^save_10/Assign_16^save_10/Assign_17^save_10/Assign_18^save_10/Assign_19^save_10/Assign_2^save_10/Assign_20^save_10/Assign_21^save_10/Assign_22^save_10/Assign_23^save_10/Assign_24^save_10/Assign_25^save_10/Assign_26^save_10/Assign_27^save_10/Assign_28^save_10/Assign_29^save_10/Assign_3^save_10/Assign_30^save_10/Assign_31^save_10/Assign_32^save_10/Assign_33^save_10/Assign_34^save_10/Assign_35^save_10/Assign_36^save_10/Assign_37^save_10/Assign_38^save_10/Assign_39^save_10/Assign_4^save_10/Assign_40^save_10/Assign_41^save_10/Assign_42^save_10/Assign_43^save_10/Assign_44^save_10/Assign_45^save_10/Assign_46^save_10/Assign_47^save_10/Assign_48^save_10/Assign_49^save_10/Assign_5^save_10/Assign_50^save_10/Assign_51^save_10/Assign_52^save_10/Assign_53^save_10/Assign_54^save_10/Assign_55^save_10/Assign_56^save_10/Assign_57^save_10/Assign_58^save_10/Assign_59^save_10/Assign_6^save_10/Assign_60^save_10/Assign_61^save_10/Assign_62^save_10/Assign_63^save_10/Assign_64^save_10/Assign_65^save_10/Assign_66^save_10/Assign_67^save_10/Assign_7^save_10/Assign_8^save_10/Assign_9
3
save_10/restore_allNoOp^save_10/restore_shard "E
save_10/Const:0save_10/Identity:0save_10/restore_all (5 @F8"çA
	variablesŮAÖA
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
~
penalty/penalty_param:0penalty/penalty_param/Assignpenalty/penalty_param/read:02%penalty/penalty_param/initial_value:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0

penalty/penalty_param/Adam:0!penalty/penalty_param/Adam/Assign!penalty/penalty_param/Adam/read:02.penalty/penalty_param/Adam/Initializer/zeros:0

penalty/penalty_param/Adam_1:0#penalty/penalty_param/Adam_1/Assign#penalty/penalty_param/Adam_1/read:020penalty/penalty_param/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
|
pi/dense/kernel/Adam:0pi/dense/kernel/Adam/Assignpi/dense/kernel/Adam/read:02(pi/dense/kernel/Adam/Initializer/zeros:0

pi/dense/kernel/Adam_1:0pi/dense/kernel/Adam_1/Assignpi/dense/kernel/Adam_1/read:02*pi/dense/kernel/Adam_1/Initializer/zeros:0
t
pi/dense/bias/Adam:0pi/dense/bias/Adam/Assignpi/dense/bias/Adam/read:02&pi/dense/bias/Adam/Initializer/zeros:0
|
pi/dense/bias/Adam_1:0pi/dense/bias/Adam_1/Assignpi/dense/bias/Adam_1/read:02(pi/dense/bias/Adam_1/Initializer/zeros:0

pi/dense_1/kernel/Adam:0pi/dense_1/kernel/Adam/Assignpi/dense_1/kernel/Adam/read:02*pi/dense_1/kernel/Adam/Initializer/zeros:0

pi/dense_1/kernel/Adam_1:0pi/dense_1/kernel/Adam_1/Assignpi/dense_1/kernel/Adam_1/read:02,pi/dense_1/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_1/bias/Adam:0pi/dense_1/bias/Adam/Assignpi/dense_1/bias/Adam/read:02(pi/dense_1/bias/Adam/Initializer/zeros:0

pi/dense_1/bias/Adam_1:0pi/dense_1/bias/Adam_1/Assignpi/dense_1/bias/Adam_1/read:02*pi/dense_1/bias/Adam_1/Initializer/zeros:0

pi/dense_2/kernel/Adam:0pi/dense_2/kernel/Adam/Assignpi/dense_2/kernel/Adam/read:02*pi/dense_2/kernel/Adam/Initializer/zeros:0

pi/dense_2/kernel/Adam_1:0pi/dense_2/kernel/Adam_1/Assignpi/dense_2/kernel/Adam_1/read:02,pi/dense_2/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_2/bias/Adam:0pi/dense_2/bias/Adam/Assignpi/dense_2/bias/Adam/read:02(pi/dense_2/bias/Adam/Initializer/zeros:0

pi/dense_2/bias/Adam_1:0pi/dense_2/bias/Adam_1/Assignpi/dense_2/bias/Adam_1/read:02*pi/dense_2/bias/Adam_1/Initializer/zeros:0
h
pi/log_std/Adam:0pi/log_std/Adam/Assignpi/log_std/Adam/read:02#pi/log_std/Adam/Initializer/zeros:0
p
pi/log_std/Adam_1:0pi/log_std/Adam_1/Assignpi/log_std/Adam_1/read:02%pi/log_std/Adam_1/Initializer/zeros:0

penalty/penalty_param/Adam_2:0#penalty/penalty_param/Adam_2/Assign#penalty/penalty_param/Adam_2/read:020penalty/penalty_param/Adam_2/Initializer/zeros:0

penalty/penalty_param/Adam_3:0#penalty/penalty_param/Adam_3/Assign#penalty/penalty_param/Adam_3/read:020penalty/penalty_param/Adam_3/Initializer/zeros:0
\
beta1_power_2:0beta1_power_2/Assignbeta1_power_2/read:02beta1_power_2/initial_value:0
\
beta2_power_2:0beta2_power_2/Assignbeta2_power_2/read:02beta2_power_2/initial_value:0
|
vf/dense/kernel/Adam:0vf/dense/kernel/Adam/Assignvf/dense/kernel/Adam/read:02(vf/dense/kernel/Adam/Initializer/zeros:0

vf/dense/kernel/Adam_1:0vf/dense/kernel/Adam_1/Assignvf/dense/kernel/Adam_1/read:02*vf/dense/kernel/Adam_1/Initializer/zeros:0
t
vf/dense/bias/Adam:0vf/dense/bias/Adam/Assignvf/dense/bias/Adam/read:02&vf/dense/bias/Adam/Initializer/zeros:0
|
vf/dense/bias/Adam_1:0vf/dense/bias/Adam_1/Assignvf/dense/bias/Adam_1/read:02(vf/dense/bias/Adam_1/Initializer/zeros:0

vf/dense_1/kernel/Adam:0vf/dense_1/kernel/Adam/Assignvf/dense_1/kernel/Adam/read:02*vf/dense_1/kernel/Adam/Initializer/zeros:0

vf/dense_1/kernel/Adam_1:0vf/dense_1/kernel/Adam_1/Assignvf/dense_1/kernel/Adam_1/read:02,vf/dense_1/kernel/Adam_1/Initializer/zeros:0
|
vf/dense_1/bias/Adam:0vf/dense_1/bias/Adam/Assignvf/dense_1/bias/Adam/read:02(vf/dense_1/bias/Adam/Initializer/zeros:0

vf/dense_1/bias/Adam_1:0vf/dense_1/bias/Adam_1/Assignvf/dense_1/bias/Adam_1/read:02*vf/dense_1/bias/Adam_1/Initializer/zeros:0

vf/dense_2/kernel/Adam:0vf/dense_2/kernel/Adam/Assignvf/dense_2/kernel/Adam/read:02*vf/dense_2/kernel/Adam/Initializer/zeros:0

vf/dense_2/kernel/Adam_1:0vf/dense_2/kernel/Adam_1/Assignvf/dense_2/kernel/Adam_1/read:02,vf/dense_2/kernel/Adam_1/Initializer/zeros:0
|
vf/dense_2/bias/Adam:0vf/dense_2/bias/Adam/Assignvf/dense_2/bias/Adam/read:02(vf/dense_2/bias/Adam/Initializer/zeros:0

vf/dense_2/bias/Adam_1:0vf/dense_2/bias/Adam_1/Assignvf/dense_2/bias/Adam_1/read:02*vf/dense_2/bias/Adam_1/Initializer/zeros:0
|
vc/dense/kernel/Adam:0vc/dense/kernel/Adam/Assignvc/dense/kernel/Adam/read:02(vc/dense/kernel/Adam/Initializer/zeros:0

vc/dense/kernel/Adam_1:0vc/dense/kernel/Adam_1/Assignvc/dense/kernel/Adam_1/read:02*vc/dense/kernel/Adam_1/Initializer/zeros:0
t
vc/dense/bias/Adam:0vc/dense/bias/Adam/Assignvc/dense/bias/Adam/read:02&vc/dense/bias/Adam/Initializer/zeros:0
|
vc/dense/bias/Adam_1:0vc/dense/bias/Adam_1/Assignvc/dense/bias/Adam_1/read:02(vc/dense/bias/Adam_1/Initializer/zeros:0

vc/dense_1/kernel/Adam:0vc/dense_1/kernel/Adam/Assignvc/dense_1/kernel/Adam/read:02*vc/dense_1/kernel/Adam/Initializer/zeros:0

vc/dense_1/kernel/Adam_1:0vc/dense_1/kernel/Adam_1/Assignvc/dense_1/kernel/Adam_1/read:02,vc/dense_1/kernel/Adam_1/Initializer/zeros:0
|
vc/dense_1/bias/Adam:0vc/dense_1/bias/Adam/Assignvc/dense_1/bias/Adam/read:02(vc/dense_1/bias/Adam/Initializer/zeros:0

vc/dense_1/bias/Adam_1:0vc/dense_1/bias/Adam_1/Assignvc/dense_1/bias/Adam_1/read:02*vc/dense_1/bias/Adam_1/Initializer/zeros:0

vc/dense_2/kernel/Adam:0vc/dense_2/kernel/Adam/Assignvc/dense_2/kernel/Adam/read:02*vc/dense_2/kernel/Adam/Initializer/zeros:0

vc/dense_2/kernel/Adam_1:0vc/dense_2/kernel/Adam_1/Assignvc/dense_2/kernel/Adam_1/read:02,vc/dense_2/kernel/Adam_1/Initializer/zeros:0
|
vc/dense_2/bias/Adam:0vc/dense_2/bias/Adam/Assignvc/dense_2/bias/Adam/read:02(vc/dense_2/bias/Adam/Initializer/zeros:0

vc/dense_2/bias/Adam_1:0vc/dense_2/bias/Adam_1/Assignvc/dense_2/bias/Adam_1/read:02*vc/dense_2/bias/Adam_1/Initializer/zeros:0"$
train_op

Adam
Adam_1
Adam_2"đ
trainable_variablesŘŐ
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
~
penalty/penalty_param:0penalty/penalty_param/Assignpenalty/penalty_param/read:02%penalty/penalty_param/initial_value:08*Ď
serving_defaultť
)
x$
Placeholder:0˙˙˙˙˙˙˙˙˙<%
pi
pi/add:0˙˙˙˙˙˙˙˙˙$
v
vf/Squeeze:0˙˙˙˙˙˙˙˙˙%
vc
vc/Squeeze:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict