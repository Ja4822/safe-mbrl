Áé0
ˇ''
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
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
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
ö
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
Ttype"serve*1.15.42v1.15.3-68-gdf8c55cî0
n
PlaceholderPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
shape:˙˙˙˙˙˙˙˙˙<*
dtype0
p
Placeholder_1Placeholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
Placeholder_4Placeholder*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0
h
Placeholder_5Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
Placeholder_6Placeholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
N
Placeholder_7Placeholder*
_output_shapes
: *
shape: *
dtype0
N
Placeholder_8Placeholder*
_output_shapes
: *
dtype0*
shape: 
Ľ
0pi/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:*
valueB"<      

.pi/dense/kernel/Initializer/random_uniform/minConst*"
_class
loc:@pi/dense/kernel*
dtype0*
_output_shapes
: *
valueB
 *ž

.pi/dense/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@pi/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *>
ď
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
T0*
seed2*"
_class
loc:@pi/dense/kernel*
dtype0*

seed *
_output_shapes
:	<
Ú
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *"
_class
loc:@pi/dense/kernel*
T0
í
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	<
ß
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	<*
T0*"
_class
loc:@pi/dense/kernel
Š
pi/dense/kernel
VariableV2*"
_class
loc:@pi/dense/kernel*
shared_name *
shape:	<*
	container *
dtype0*
_output_shapes
:	<
Ô
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
T0

pi/dense/kernel/readIdentitypi/dense/kernel*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<

pi/dense/bias/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB*    *
_output_shapes	
:*
dtype0

pi/dense/bias
VariableV2*
shape:* 
_class
loc:@pi/dense/bias*
shared_name *
dtype0*
	container *
_output_shapes	
:
ż
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
use_locking(*
T0
u
pi/dense/bias/readIdentitypi/dense/bias*
_output_shapes	
:*
T0* 
_class
loc:@pi/dense/bias

pi/dense/MatMulMatMulPlaceholderpi/dense/kernel/read*
T0*
transpose_b( *
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
Z
pi/dense/TanhTanhpi/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:*$
_class
loc:@pi/dense_1/kernel

0pi/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *×łÝ˝*
dtype0*
_output_shapes
: *$
_class
loc:@pi/dense_1/kernel

0pi/dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *×łÝ=*$
_class
loc:@pi/dense_1/kernel*
dtype0
ö
:pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_1/kernel/Initializer/random_uniform/shape*$
_class
loc:@pi/dense_1/kernel*
seed2* 
_output_shapes
:
*

seed *
dtype0*
T0
â
0pi/dense_1/kernel/Initializer/random_uniform/subSub0pi/dense_1/kernel/Initializer/random_uniform/max0pi/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *$
_class
loc:@pi/dense_1/kernel*
T0
ö
0pi/dense_1/kernel/Initializer/random_uniform/mulMul:pi/dense_1/kernel/Initializer/random_uniform/RandomUniform0pi/dense_1/kernel/Initializer/random_uniform/sub*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
T0
č
,pi/dense_1/kernel/Initializer/random_uniformAdd0pi/dense_1/kernel/Initializer/random_uniform/mul0pi/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:

Ż
pi/dense_1/kernel
VariableV2* 
_output_shapes
:
*
dtype0*
	container *$
_class
loc:@pi/dense_1/kernel*
shared_name *
shape:

Ý
pi/dense_1/kernel/AssignAssignpi/dense_1/kernel,pi/dense_1/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel

pi/dense_1/kernel/readIdentitypi/dense_1/kernel*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel

!pi/dense_1/bias/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias
Ą
pi/dense_1/bias
VariableV2*"
_class
loc:@pi/dense_1/bias*
shared_name *
shape:*
	container *
dtype0*
_output_shapes	
:
Ç
pi/dense_1/bias/AssignAssignpi/dense_1/bias!pi/dense_1/bias/Initializer/zeros*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(*"
_class
loc:@pi/dense_1/bias
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
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

pi/dense_1/BiasAddBiasAddpi/dense_1/MatMulpi/dense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
^
pi/dense_1/TanhTanhpi/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *$
_class
loc:@pi/dense_2/kernel*
dtype0*
_output_shapes
:

0pi/dense_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *$
_class
loc:@pi/dense_2/kernel*
valueB
 *(ž*
dtype0

0pi/dense_2/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *$
_class
loc:@pi/dense_2/kernel*
dtype0*
valueB
 *(>
ő
:pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_2/kernel/Initializer/random_uniform/shape*

seed *
seed2.*
dtype0*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
:	
â
0pi/dense_2/kernel/Initializer/random_uniform/subSub0pi/dense_2/kernel/Initializer/random_uniform/max0pi/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
: 
ő
0pi/dense_2/kernel/Initializer/random_uniform/mulMul:pi/dense_2/kernel/Initializer/random_uniform/RandomUniform0pi/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0
ç
,pi/dense_2/kernel/Initializer/random_uniformAdd0pi/dense_2/kernel/Initializer/random_uniform/mul0pi/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
:	
­
pi/dense_2/kernel
VariableV2*
	container *$
_class
loc:@pi/dense_2/kernel*
dtype0*
_output_shapes
:	*
shape:	*
shared_name 
Ü
pi/dense_2/kernel/AssignAssignpi/dense_2/kernel,pi/dense_2/kernel/Initializer/random_uniform*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(

pi/dense_2/kernel/readIdentitypi/dense_2/kernel*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0

!pi/dense_2/bias/Initializer/zerosConst*
_output_shapes
:*
dtype0*
valueB*    *"
_class
loc:@pi/dense_2/bias

pi/dense_2/bias
VariableV2*
shape:*
dtype0*
	container *"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
shared_name 
Ć
pi/dense_2/bias/AssignAssignpi/dense_2/bias!pi/dense_2/bias/Initializer/zeros*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
use_locking(
z
pi/dense_2/bias/readIdentitypi/dense_2/bias*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias

pi/dense_2/MatMulMatMulpi/dense_1/Tanhpi/dense_2/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0*
transpose_b( 

pi/dense_2/BiasAddBiasAddpi/dense_2/MatMulpi/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
i
pi/log_std/initial_valueConst*
dtype0*
valueB"   ż   ż*
_output_shapes
:
v

pi/log_std
VariableV2*
shape:*
_output_shapes
:*
dtype0*
shared_name *
	container 
Ž
pi/log_std/AssignAssign
pi/log_stdpi/log_std/initial_value*
T0*
validate_shape(*
_class
loc:@pi/log_std*
use_locking(*
_output_shapes
:
k
pi/log_std/readIdentity
pi/log_std*
_output_shapes
:*
_class
loc:@pi/log_std*
T0
C
pi/ExpExppi/log_std/read*
_output_shapes
:*
T0
Z
pi/ShapeShapepi/dense_2/BiasAdd*
T0*
_output_shapes
:*
out_type0
Z
pi/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
pi/random_normal/stddevConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 

%pi/random_normal/RandomStandardNormalRandomStandardNormalpi/Shape*
seed2C*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

seed *
dtype0*
T0

pi/random_normal/mulMul%pi/random_normal/RandomStandardNormalpi/random_normal/stddev*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
pi/random_normalAddpi/random_normal/mulpi/random_normal/mean*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Y
pi/mulMulpi/random_normalpi/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
pi/addAddV2pi/dense_2/BiasAddpi/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
pi/pow/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0
U
pi/powPow
pi/truedivpi/pow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_1/xConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
Q
pi/mul_1Mul
pi/mul_1/xpi/log_std/read*
T0*
_output_shapes
:
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
pi/add_3/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_2/xConst*
dtype0*
valueB
 *   ż*
_output_shapes
: 
W
pi/mul_2Mul
pi/mul_2/xpi/add_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
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
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0
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

pi/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
L
pi/add_4AddV2pi/Exp_2
pi/add_4/y*
_output_shapes
:*
T0
]
pi/truediv_1RealDivpi/sub_1pi/add_4*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/pow_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
[
pi/pow_1Powpi/truediv_1
pi/pow_1/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/mul_3/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
Q
pi/mul_3Mul
pi/mul_3/xpi/log_std/read*
T0*
_output_shapes
:
W
pi/add_5AddV2pi/pow_1pi/mul_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/add_6/yConst*
valueB
 *?ë?*
dtype0*
_output_shapes
: 
Y
pi/add_6AddV2pi/add_5
pi/add_6/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/mul_4/xConst*
dtype0*
_output_shapes
: *
valueB
 *   ż
W
pi/mul_4Mul
pi/mul_4/xpi/add_6*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
pi/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :

pi/Sum_1Sumpi/mul_4pi/Sum_1/reduction_indices*

Tidx0*
	keep_dims( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
q
pi/PlaceholderPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
pi/Placeholder_1Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_5/xConst*
valueB
 *   @*
_output_shapes
: *
dtype0
Q
pi/mul_5Mul
pi/mul_5/xpi/log_std/read*
T0*
_output_shapes
:
>
pi/Exp_3Exppi/mul_5*
T0*
_output_shapes
:
O

pi/mul_6/xConst*
_output_shapes
: *
valueB
 *   @*
dtype0
_
pi/mul_6Mul
pi/mul_6/xpi/Placeholder_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
K
pi/Exp_4Exppi/mul_6*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
e
pi/sub_2Subpi/Placeholderpi/dense_2/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/pow_2/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
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

pi/add_8/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
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

pi/sub_3/yConst*
_output_shapes
: *
valueB
 *  ?*
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
: *
valueB
 *   ?*
dtype0
W
pi/mul_7Mul
pi/mul_7/xpi/sub_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
_
pi/add_9AddV2pi/mul_7pi/Placeholder_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
\
pi/sub_4Subpi/add_9pi/log_std/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
\
pi/Sum_2/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :

pi/Sum_2Sumpi/sub_4pi/Sum_2/reduction_indices*
	keep_dims( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0
R
pi/ConstConst*
_output_shapes
:*
valueB: *
dtype0
a
pi/MeanMeanpi/Sum_2pi/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
P
pi/add_10/yConst*
dtype0*
valueB
 *Çľ?*
_output_shapes
: 
U
	pi/add_10AddV2pi/log_std/readpi/add_10/y*
T0*
_output_shapes
:
e
pi/Sum_3/reduction_indicesConst*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: 
t
pi/Sum_3Sum	pi/add_10pi/Sum_3/reduction_indices*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
M

pi/Const_1Const*
dtype0*
_output_shapes
: *
valueB 
e
	pi/Mean_1Meanpi/Sum_3
pi/Const_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
Ľ
0vf/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"<      *"
_class
loc:@vf/dense/kernel*
dtype0*
_output_shapes
:

.vf/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *ž*
dtype0*"
_class
loc:@vf/dense/kernel*
_output_shapes
: 

.vf/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@vf/dense/kernel*
valueB
 *>*
dtype0
đ
8vf/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vf/dense/kernel/Initializer/random_uniform/shape*"
_class
loc:@vf/dense/kernel*
T0*

seed *
_output_shapes
:	<*
dtype0*
seed2
Ú
.vf/dense/kernel/Initializer/random_uniform/subSub.vf/dense/kernel/Initializer/random_uniform/max.vf/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
: 
í
.vf/dense/kernel/Initializer/random_uniform/mulMul8vf/dense/kernel/Initializer/random_uniform/RandomUniform.vf/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
T0
ß
*vf/dense/kernel/Initializer/random_uniformAdd.vf/dense/kernel/Initializer/random_uniform/mul.vf/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	<
Š
vf/dense/kernel
VariableV2*
shape:	<*"
_class
loc:@vf/dense/kernel*
dtype0*
shared_name *
	container *
_output_shapes
:	<
Ô
vf/dense/kernel/AssignAssignvf/dense/kernel*vf/dense/kernel/Initializer/random_uniform*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	<

vf/dense/kernel/readIdentityvf/dense/kernel*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
T0

vf/dense/bias/Initializer/zerosConst*
valueB*    * 
_class
loc:@vf/dense/bias*
dtype0*
_output_shapes	
:

vf/dense/bias
VariableV2*
shape:*
shared_name *
_output_shapes	
:*
dtype0*
	container * 
_class
loc:@vf/dense/bias
ż
vf/dense/bias/AssignAssignvf/dense/biasvf/dense/bias/Initializer/zeros*
T0*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
use_locking(
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
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( *
transpose_b( 

vf/dense/BiasAddBiasAddvf/dense/MatMulvf/dense/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
Z
vf/dense/TanhTanhvf/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2vf/dense_1/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@vf/dense_1/kernel*
valueB"      *
_output_shapes
:*
dtype0

0vf/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *×łÝ˝*
_output_shapes
: *$
_class
loc:@vf/dense_1/kernel

0vf/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *×łÝ=*$
_class
loc:@vf/dense_1/kernel*
dtype0*
_output_shapes
: 
÷
:vf/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*
seed2*
T0*

seed * 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
â
0vf/dense_1/kernel/Initializer/random_uniform/subSub0vf/dense_1/kernel/Initializer/random_uniform/max0vf/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@vf/dense_1/kernel
ö
0vf/dense_1/kernel/Initializer/random_uniform/mulMul:vf/dense_1/kernel/Initializer/random_uniform/RandomUniform0vf/dense_1/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0
č
,vf/dense_1/kernel/Initializer/random_uniformAdd0vf/dense_1/kernel/Initializer/random_uniform/mul0vf/dense_1/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
Ż
vf/dense_1/kernel
VariableV2*
dtype0*
shared_name *
	container *$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
shape:

Ý
vf/dense_1/kernel/AssignAssignvf/dense_1/kernel,vf/dense_1/kernel/Initializer/random_uniform*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:
*
use_locking(

vf/dense_1/kernel/readIdentityvf/dense_1/kernel*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:


!vf/dense_1/bias/Initializer/zerosConst*"
_class
loc:@vf/dense_1/bias*
valueB*    *
_output_shapes	
:*
dtype0
Ą
vf/dense_1/bias
VariableV2*
shape:*
dtype0*
	container *
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
shared_name 
Ç
vf/dense_1/bias/AssignAssignvf/dense_1/bias!vf/dense_1/bias/Initializer/zeros*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0*
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
vf/dense_1/MatMulMatMulvf/dense/Tanhvf/dense_1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

vf/dense_1/BiasAddBiasAddvf/dense_1/MatMulvf/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
vf/dense_1/TanhTanhvf/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2vf/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *$
_class
loc:@vf/dense_2/kernel

0vf/dense_2/kernel/Initializer/random_uniform/minConst*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
: *
valueB
 *Ivž*
dtype0

0vf/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *Iv>*
_output_shapes
: *$
_class
loc:@vf/dense_2/kernel*
dtype0
ö
:vf/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_2/kernel/Initializer/random_uniform/shape*
_output_shapes
:	*
dtype0*$
_class
loc:@vf/dense_2/kernel*
seed2Ź*
T0*

seed 
â
0vf/dense_2/kernel/Initializer/random_uniform/subSub0vf/dense_2/kernel/Initializer/random_uniform/max0vf/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@vf/dense_2/kernel
ő
0vf/dense_2/kernel/Initializer/random_uniform/mulMul:vf/dense_2/kernel/Initializer/random_uniform/RandomUniform0vf/dense_2/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	*$
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
VariableV2*
shared_name *
	container *$
_class
loc:@vf/dense_2/kernel*
shape:	*
_output_shapes
:	*
dtype0
Ü
vf/dense_2/kernel/AssignAssignvf/dense_2/kernel,vf/dense_2/kernel/Initializer/random_uniform*
T0*
validate_shape(*
_output_shapes
:	*
use_locking(*$
_class
loc:@vf/dense_2/kernel

vf/dense_2/kernel/readIdentityvf/dense_2/kernel*
_output_shapes
:	*
T0*$
_class
loc:@vf/dense_2/kernel

!vf/dense_2/bias/Initializer/zerosConst*
dtype0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
valueB*    

vf/dense_2/bias
VariableV2*
dtype0*"
_class
loc:@vf/dense_2/bias*
	container *
_output_shapes
:*
shared_name *
shape:
Ć
vf/dense_2/bias/AssignAssignvf/dense_2/bias!vf/dense_2/bias/Initializer/zeros*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
z
vf/dense_2/bias/readIdentityvf/dense_2/bias*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias

vf/dense_2/MatMulMatMulvf/dense_1/Tanhvf/dense_2/kernel/read*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( 

vf/dense_2/BiasAddBiasAddvf/dense_2/MatMulvf/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
n

vf/SqueezeSqueezevf/dense_2/BiasAdd*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims

Ľ
0vc/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"<      *
dtype0*
_output_shapes
:*"
_class
loc:@vc/dense/kernel

.vc/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *ž*"
_class
loc:@vc/dense/kernel*
dtype0

.vc/dense/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@vc/dense/kernel*
_output_shapes
: *
valueB
 *>*
dtype0
đ
8vc/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vc/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	<*

seed *
T0*
seed2˝*"
_class
loc:@vc/dense/kernel
Ú
.vc/dense/kernel/Initializer/random_uniform/subSub.vc/dense/kernel/Initializer/random_uniform/max.vc/dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *"
_class
loc:@vc/dense/kernel
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
VariableV2*
shape:	<*"
_class
loc:@vc/dense/kernel*
shared_name *
dtype0*
_output_shapes
:	<*
	container 
Ô
vc/dense/kernel/AssignAssignvc/dense/kernel*vc/dense/kernel/Initializer/random_uniform*
use_locking(*
_output_shapes
:	<*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel

vc/dense/kernel/readIdentityvc/dense/kernel*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<

vc/dense/bias/Initializer/zerosConst*
valueB*    * 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
dtype0

vc/dense/bias
VariableV2*
shape:*
shared_name * 
_class
loc:@vc/dense/bias*
dtype0*
_output_shapes	
:*
	container 
ż
vc/dense/bias/AssignAssignvc/dense/biasvc/dense/bias/Initializer/zeros* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(
u
vc/dense/bias/readIdentityvc/dense/bias*
_output_shapes	
:*
T0* 
_class
loc:@vc/dense/bias

vc/dense/MatMulMatMulPlaceholdervc/dense/kernel/read*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( 

vc/dense/BiasAddBiasAddvc/dense/MatMulvc/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
vc/dense/TanhTanhvc/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
2vc/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *$
_class
loc:@vc/dense_1/kernel*
dtype0

0vc/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *×łÝ˝*$
_class
loc:@vc/dense_1/kernel

0vc/dense_1/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@vc/dense_1/kernel*
valueB
 *×łÝ=*
dtype0*
_output_shapes
: 
÷
:vc/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*
T0*
seed2Î*

seed *$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

â
0vc/dense_1/kernel/Initializer/random_uniform/subSub0vc/dense_1/kernel/Initializer/random_uniform/max0vc/dense_1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@vc/dense_1/kernel*
_output_shapes
: 
ö
0vc/dense_1/kernel/Initializer/random_uniform/mulMul:vc/dense_1/kernel/Initializer/random_uniform/RandomUniform0vc/dense_1/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

č
,vc/dense_1/kernel/Initializer/random_uniformAdd0vc/dense_1/kernel/Initializer/random_uniform/mul0vc/dense_1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

Ż
vc/dense_1/kernel
VariableV2*
dtype0*
shape:
*$
_class
loc:@vc/dense_1/kernel*
	container *
shared_name * 
_output_shapes
:

Ý
vc/dense_1/kernel/AssignAssignvc/dense_1/kernel,vc/dense_1/kernel/Initializer/random_uniform*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:


vc/dense_1/kernel/readIdentityvc/dense_1/kernel*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:


!vc/dense_1/bias/Initializer/zerosConst*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
dtype0*
valueB*    
Ą
vc/dense_1/bias
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
Ç
vc/dense_1/bias/AssignAssignvc/dense_1/bias!vc/dense_1/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias
{
vc/dense_1/bias/readIdentityvc/dense_1/bias*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0

vc/dense_1/MatMulMatMulvc/dense/Tanhvc/dense_1/kernel/read*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( 

vc/dense_1/BiasAddBiasAddvc/dense_1/MatMulvc/dense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
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
0vc/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *Ivž*
dtype0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
: 

0vc/dense_2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *Iv>*$
_class
loc:@vc/dense_2/kernel
ö
:vc/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_2/kernel/Initializer/random_uniform/shape*$
_class
loc:@vc/dense_2/kernel*
T0*
dtype0*

seed *
seed2ß*
_output_shapes
:	
â
0vc/dense_2/kernel/Initializer/random_uniform/subSub0vc/dense_2/kernel/Initializer/random_uniform/max0vc/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@vc/dense_2/kernel
ő
0vc/dense_2/kernel/Initializer/random_uniform/mulMul:vc/dense_2/kernel/Initializer/random_uniform/RandomUniform0vc/dense_2/kernel/Initializer/random_uniform/sub*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	
ç
,vc/dense_2/kernel/Initializer/random_uniformAdd0vc/dense_2/kernel/Initializer/random_uniform/mul0vc/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
T0
­
vc/dense_2/kernel
VariableV2*
shared_name *
_output_shapes
:	*
	container *
shape:	*$
_class
loc:@vc/dense_2/kernel*
dtype0
Ü
vc/dense_2/kernel/AssignAssignvc/dense_2/kernel,vc/dense_2/kernel/Initializer/random_uniform*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0

vc/dense_2/kernel/readIdentityvc/dense_2/kernel*
T0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel

!vc/dense_2/bias/Initializer/zerosConst*
valueB*    *
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
dtype0

vc/dense_2/bias
VariableV2*
shared_name *
_output_shapes
:*
dtype0*
shape:*"
_class
loc:@vc/dense_2/bias*
	container 
Ć
vc/dense_2/bias/AssignAssignvc/dense_2/bias!vc/dense_2/bias/Initializer/zeros*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(
z
vc/dense_2/bias/readIdentityvc/dense_2/bias*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0

vc/dense_2/MatMulMatMulvc/dense_1/Tanhvc/dense_2/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
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
NegNegpi/Sum*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
O
ConstConst*
dtype0*
_output_shapes
:*
valueB: 
V
MeanMeanNegConst*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0
O
subSubpi/SumPlaceholder_6*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
=
ExpExpsub*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
mulMulExpPlaceholder_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Z
Mean_1MeanmulConst_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
N
mul_1MulExpPlaceholder_3*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_2Const*
dtype0*
valueB: *
_output_shapes
:
\
Mean_2Meanmul_1Const_2*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
L
mul_2/xConst*
_output_shapes
: *
valueB
 *    *
dtype0
A
mul_2Mulmul_2/x	pi/Mean_1*
T0*
_output_shapes
: 
<
addAddV2Mean_1mul_2*
T0*
_output_shapes
: 
2
Neg_1Negadd*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
P
gradients/Neg_1_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
m
#gradients/Mean_1_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_1_grad/ReshapeReshapegradients/Neg_1_grad/Neg#gradients/Mean_1_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
^
gradients/Mean_1_grad/ShapeShapemul*
_output_shapes
:*
T0*
out_type0

gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
gradients/Mean_1_grad/Shape_1Shapemul*
_output_shapes
:*
T0*
out_type0
`
gradients/Mean_1_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
e
gradients/Mean_1_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_1gradients/Mean_1_grad/Const*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
g
gradients/Mean_1_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
 
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
a
gradients/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0

gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0

gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
_output_shapes
: *
T0

gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*
_output_shapes
: *
Truncate( *

DstT0*

SrcT0

gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
T0*
_output_shapes
:*
out_type0
g
gradients/mul_grad/Shape_1ShapePlaceholder_2*
out_type0*
T0*
_output_shapes
:
´
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
y
gradients/mul_grad/MulMulgradients/Mean_1_grad/truedivPlaceholder_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
q
gradients/mul_grad/Mul_1MulExpgradients/Mean_1_grad/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
&gradients/pi/Mean_1_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0

 gradients/pi/Mean_1_grad/ReshapeReshapegradients/mul_2_grad/Mul_1&gradients/pi/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
a
gradients/pi/Mean_1_grad/ConstConst*
dtype0*
_output_shapes
: *
valueB 

gradients/pi/Mean_1_grad/TileTile gradients/pi/Mean_1_grad/Reshapegradients/pi/Mean_1_grad/Const*
_output_shapes
: *
T0*

Tmultiples0
e
 gradients/pi/Mean_1_grad/Const_1Const*
valueB
 *  ?*
_output_shapes
: *
dtype0

 gradients/pi/Mean_1_grad/truedivRealDivgradients/pi/Mean_1_grad/Tile gradients/pi/Mean_1_grad/Const_1*
_output_shapes
: *
T0
l
gradients/Exp_grad/mulMulgradients/mul_grad/ReshapeExp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
gradients/pi/Sum_3_grad/Cast/xConst*
dtype0*
_output_shapes
:*
valueB:
s
 gradients/pi/Sum_3_grad/Cast_1/xConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
^
gradients/pi/Sum_3_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :

gradients/pi/Sum_3_grad/addAddV2 gradients/pi/Sum_3_grad/Cast_1/xgradients/pi/Sum_3_grad/Size*
T0*
_output_shapes
:

gradients/pi/Sum_3_grad/modFloorModgradients/pi/Sum_3_grad/addgradients/pi/Sum_3_grad/Size*
_output_shapes
:*
T0
g
gradients/pi/Sum_3_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0
e
#gradients/pi/Sum_3_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0
e
#gradients/pi/Sum_3_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ś
gradients/pi/Sum_3_grad/rangeRange#gradients/pi/Sum_3_grad/range/startgradients/pi/Sum_3_grad/Size#gradients/pi/Sum_3_grad/range/delta*

Tidx0*
_output_shapes
:
d
"gradients/pi/Sum_3_grad/Fill/valueConst*
_output_shapes
: *
value	B :*
dtype0

gradients/pi/Sum_3_grad/FillFillgradients/pi/Sum_3_grad/Shape"gradients/pi/Sum_3_grad/Fill/value*

index_type0*
_output_shapes
:*
T0
Ţ
%gradients/pi/Sum_3_grad/DynamicStitchDynamicStitchgradients/pi/Sum_3_grad/rangegradients/pi/Sum_3_grad/modgradients/pi/Sum_3_grad/Cast/xgradients/pi/Sum_3_grad/Fill*
T0*
_output_shapes
:*
N
k
!gradients/pi/Sum_3_grad/Maximum/xConst*
dtype0*
valueB:*
_output_shapes
:
c
!gradients/pi/Sum_3_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0

gradients/pi/Sum_3_grad/MaximumMaximum!gradients/pi/Sum_3_grad/Maximum/x!gradients/pi/Sum_3_grad/Maximum/y*
T0*
_output_shapes
:
l
"gradients/pi/Sum_3_grad/floordiv/xConst*
valueB:*
_output_shapes
:*
dtype0

 gradients/pi/Sum_3_grad/floordivFloorDiv"gradients/pi/Sum_3_grad/floordiv/xgradients/pi/Sum_3_grad/Maximum*
T0*
_output_shapes
:
o
%gradients/pi/Sum_3_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
Ś
gradients/pi/Sum_3_grad/ReshapeReshape gradients/pi/Mean_1_grad/truediv%gradients/pi/Sum_3_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
p
&gradients/pi/Sum_3_grad/Tile/multiplesConst*
_output_shapes
:*
valueB:*
dtype0
¤
gradients/pi/Sum_3_grad/TileTilegradients/pi/Sum_3_grad/Reshape&gradients/pi/Sum_3_grad/Tile/multiples*
T0*
_output_shapes
:*

Tmultiples0
^
gradients/sub_grad/ShapeShapepi/Sum*
T0*
_output_shapes
:*
out_type0
g
gradients/sub_grad/Shape_1ShapePlaceholder_6*
_output_shapes
:*
T0*
out_type0
´
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/sub_grad/SumSumgradients/Exp_grad/mul(gradients/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
c
gradients/sub_grad/NegNeggradients/Exp_grad/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
gradients/sub_grad/Sum_1Sumgradients/sub_grad/Neg*gradients/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Sum_1gradients/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
{
1gradients/pi/add_10_grad/BroadcastGradientArgs/s0Const*
_output_shapes
:*
valueB:*
dtype0
t
1gradients/pi/add_10_grad/BroadcastGradientArgs/s1Const*
dtype0*
_output_shapes
: *
valueB 
ę
.gradients/pi/add_10_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/pi/add_10_grad/BroadcastGradientArgs/s01gradients/pi/add_10_grad/BroadcastGradientArgs/s1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
x
.gradients/pi/add_10_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
Ż
gradients/pi/add_10_grad/SumSumgradients/pi/Sum_3_grad/Tile.gradients/pi/add_10_grad/Sum/reduction_indices*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
i
&gradients/pi/add_10_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *
valueB 
 
 gradients/pi/add_10_grad/ReshapeReshapegradients/pi/add_10_grad/Sum&gradients/pi/add_10_grad/Reshape/shape*
_output_shapes
: *
Tshape0*
T0
c
gradients/pi/Sum_grad/ShapeShapepi/mul_2*
_output_shapes
:*
T0*
out_type0

gradients/pi/Sum_grad/SizeConst*
value	B :*
_output_shapes
: *
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
Š
gradients/pi/Sum_grad/addAddV2pi/Sum/reduction_indicesgradients/pi/Sum_grad/Size*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
T0*
_output_shapes
: 
­
gradients/pi/Sum_grad/modFloorModgradients/pi/Sum_grad/addgradients/pi/Sum_grad/Size*
T0*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape

gradients/pi/Sum_grad/Shape_1Const*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
valueB *
dtype0

!gradients/pi/Sum_grad/range/startConst*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
value	B : *
dtype0

!gradients/pi/Sum_grad/range/deltaConst*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
Ţ
gradients/pi/Sum_grad/rangeRange!gradients/pi/Sum_grad/range/startgradients/pi/Sum_grad/Size!gradients/pi/Sum_grad/range/delta*

Tidx0*
_output_shapes
:*.
_class$
" loc:@gradients/pi/Sum_grad/Shape

 gradients/pi/Sum_grad/Fill/valueConst*
_output_shapes
: *
value	B :*
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
Ć
gradients/pi/Sum_grad/FillFillgradients/pi/Sum_grad/Shape_1 gradients/pi/Sum_grad/Fill/value*
_output_shapes
: *
T0*

index_type0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape

#gradients/pi/Sum_grad/DynamicStitchDynamicStitchgradients/pi/Sum_grad/rangegradients/pi/Sum_grad/modgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Fill*
_output_shapes
:*
T0*
N*.
_class$
" loc:@gradients/pi/Sum_grad/Shape

gradients/pi/Sum_grad/Maximum/yConst*
value	B :*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0
Ă
gradients/pi/Sum_grad/MaximumMaximum#gradients/pi/Sum_grad/DynamicStitchgradients/pi/Sum_grad/Maximum/y*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:
ť
gradients/pi/Sum_grad/floordivFloorDivgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Maximum*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:*
T0
˛
gradients/pi/Sum_grad/ReshapeReshapegradients/sub_grad/Reshape#gradients/pi/Sum_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ľ
gradients/pi/Sum_grad/TileTilegradients/pi/Sum_grad/Reshapegradients/pi/Sum_grad/floordiv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0
e
gradients/pi/mul_2_grad/ShapeShape
pi/mul_2/x*
out_type0*
T0*
_output_shapes
: 
g
gradients/pi/mul_2_grad/Shape_1Shapepi/add_3*
_output_shapes
:*
out_type0*
T0
Ă
-gradients/pi/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/mul_2_grad/Shapegradients/pi/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
z
gradients/pi/mul_2_grad/MulMulgradients/pi/Sum_grad/Tilepi/add_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
gradients/pi/mul_2_grad/SumSumgradients/pi/mul_2_grad/Mul-gradients/pi/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

gradients/pi/mul_2_grad/ReshapeReshapegradients/pi/mul_2_grad/Sumgradients/pi/mul_2_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
~
gradients/pi/mul_2_grad/Mul_1Mul
pi/mul_2/xgradients/pi/Sum_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients/pi/mul_2_grad/Sum_1Sumgradients/pi/mul_2_grad/Mul_1/gradients/pi/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
Ź
!gradients/pi/mul_2_grad/Reshape_1Reshapegradients/pi/mul_2_grad/Sum_1gradients/pi/mul_2_grad/Shape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
e
gradients/pi/add_3_grad/ShapeShapepi/add_2*
out_type0*
T0*
_output_shapes
:
g
gradients/pi/add_3_grad/Shape_1Shape
pi/add_3/y*
out_type0*
_output_shapes
: *
T0
Ă
-gradients/pi/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_3_grad/Shapegradients/pi/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
´
gradients/pi/add_3_grad/SumSum!gradients/pi/mul_2_grad/Reshape_1-gradients/pi/add_3_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ś
gradients/pi/add_3_grad/ReshapeReshapegradients/pi/add_3_grad/Sumgradients/pi/add_3_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
¸
gradients/pi/add_3_grad/Sum_1Sum!gradients/pi/mul_2_grad/Reshape_1/gradients/pi/add_3_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0

!gradients/pi/add_3_grad/Reshape_1Reshapegradients/pi/add_3_grad/Sum_1gradients/pi/add_3_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
c
gradients/pi/add_2_grad/ShapeShapepi/pow*
_output_shapes
:*
T0*
out_type0
g
gradients/pi/add_2_grad/Shape_1Shapepi/mul_1*
_output_shapes
:*
out_type0*
T0
Ă
-gradients/pi/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_2_grad/Shapegradients/pi/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
˛
gradients/pi/add_2_grad/SumSumgradients/pi/add_3_grad/Reshape-gradients/pi/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
Ś
gradients/pi/add_2_grad/ReshapeReshapegradients/pi/add_2_grad/Sumgradients/pi/add_2_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ś
gradients/pi/add_2_grad/Sum_1Sumgradients/pi/add_3_grad/Reshape/gradients/pi/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 

!gradients/pi/add_2_grad/Reshape_1Reshapegradients/pi/add_2_grad/Sum_1gradients/pi/add_2_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
e
gradients/pi/pow_grad/ShapeShape
pi/truediv*
_output_shapes
:*
out_type0*
T0
c
gradients/pi/pow_grad/Shape_1Shapepi/pow/y*
T0*
_output_shapes
: *
out_type0
˝
+gradients/pi/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/pow_grad/Shapegradients/pi/pow_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
}
gradients/pi/pow_grad/mulMulgradients/pi/add_2_grad/Reshapepi/pow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
gradients/pi/pow_grad/sub/yConst*
valueB
 *  ?*
_output_shapes
: *
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
:˙˙˙˙˙˙˙˙˙*
T0

gradients/pi/pow_grad/mul_1Mulgradients/pi/pow_grad/mulgradients/pi/pow_grad/Pow*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ş
gradients/pi/pow_grad/SumSumgradients/pi/pow_grad/mul_1+gradients/pi/pow_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
 
gradients/pi/pow_grad/ReshapeReshapegradients/pi/pow_grad/Sumgradients/pi/pow_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
d
gradients/pi/pow_grad/Greater/yConst*
valueB
 *    *
_output_shapes
: *
dtype0

gradients/pi/pow_grad/GreaterGreater
pi/truedivgradients/pi/pow_grad/Greater/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
o
%gradients/pi/pow_grad/ones_like/ShapeShape
pi/truediv*
T0*
out_type0*
_output_shapes
:
j
%gradients/pi/pow_grad/ones_like/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
š
gradients/pi/pow_grad/ones_likeFill%gradients/pi/pow_grad/ones_like/Shape%gradients/pi/pow_grad/ones_like/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

index_type0
¤
gradients/pi/pow_grad/SelectSelectgradients/pi/pow_grad/Greater
pi/truedivgradients/pi/pow_grad/ones_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
gradients/pi/pow_grad/LogLoggradients/pi/pow_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
k
 gradients/pi/pow_grad/zeros_like	ZerosLike
pi/truediv*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
gradients/pi/pow_grad/Select_1Selectgradients/pi/pow_grad/Greatergradients/pi/pow_grad/Log gradients/pi/pow_grad/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
gradients/pi/pow_grad/mul_2Mulgradients/pi/add_2_grad/Reshapepi/pow*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/pi/pow_grad/mul_3Mulgradients/pi/pow_grad/mul_2gradients/pi/pow_grad/Select_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
gradients/pi/pow_grad/Sum_1Sumgradients/pi/pow_grad/mul_3-gradients/pi/pow_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/pi/pow_grad/Reshape_1Reshapegradients/pi/pow_grad/Sum_1gradients/pi/pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
s
0gradients/pi/mul_1_grad/BroadcastGradientArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 
z
0gradients/pi/mul_1_grad/BroadcastGradientArgs/s1Const*
dtype0*
valueB:*
_output_shapes
:
ç
-gradients/pi/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients/pi/mul_1_grad/BroadcastGradientArgs/s00gradients/pi/mul_1_grad/BroadcastGradientArgs/s1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
{
gradients/pi/mul_1_grad/MulMul!gradients/pi/add_2_grad/Reshape_1pi/log_std/read*
T0*
_output_shapes
:
w
-gradients/pi/mul_1_grad/Sum/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
Ź
gradients/pi/mul_1_grad/SumSumgradients/pi/mul_1_grad/Mul-gradients/pi/mul_1_grad/Sum/reduction_indices*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
h
%gradients/pi/mul_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 

gradients/pi/mul_1_grad/ReshapeReshapegradients/pi/mul_1_grad/Sum%gradients/pi/mul_1_grad/Reshape/shape*
T0*
_output_shapes
: *
Tshape0
x
gradients/pi/mul_1_grad/Mul_1Mul
pi/mul_1/x!gradients/pi/add_2_grad/Reshape_1*
T0*
_output_shapes
:
e
gradients/pi/truediv_grad/ShapeShapepi/sub*
T0*
out_type0*
_output_shapes
:
k
!gradients/pi/truediv_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
É
/gradients/pi/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/truediv_grad/Shape!gradients/pi/truediv_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

!gradients/pi/truediv_grad/RealDivRealDivgradients/pi/pow_grad/Reshapepi/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
gradients/pi/truediv_grad/SumSum!gradients/pi/truediv_grad/RealDiv/gradients/pi/truediv_grad/BroadcastGradientArgs*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
Ź
!gradients/pi/truediv_grad/ReshapeReshapegradients/pi/truediv_grad/Sumgradients/pi/truediv_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
^
gradients/pi/truediv_grad/NegNegpi/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

#gradients/pi/truediv_grad/RealDiv_1RealDivgradients/pi/truediv_grad/Negpi/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

#gradients/pi/truediv_grad/RealDiv_2RealDiv#gradients/pi/truediv_grad/RealDiv_1pi/add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/pi/truediv_grad/mulMulgradients/pi/pow_grad/Reshape#gradients/pi/truediv_grad/RealDiv_2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
gradients/pi/truediv_grad/Sum_1Sumgradients/pi/truediv_grad/mul1gradients/pi/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Ľ
#gradients/pi/truediv_grad/Reshape_1Reshapegradients/pi/truediv_grad/Sum_1!gradients/pi/truediv_grad/Shape_1*
T0*
_output_shapes
:*
Tshape0
h
gradients/pi/sub_grad/ShapeShapePlaceholder_1*
out_type0*
T0*
_output_shapes
:
o
gradients/pi/sub_grad/Shape_1Shapepi/dense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
˝
+gradients/pi/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/sub_grad/Shapegradients/pi/sub_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
°
gradients/pi/sub_grad/SumSum!gradients/pi/truediv_grad/Reshape+gradients/pi/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
 
gradients/pi/sub_grad/ReshapeReshapegradients/pi/sub_grad/Sumgradients/pi/sub_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
u
gradients/pi/sub_grad/NegNeg!gradients/pi/truediv_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
gradients/pi/sub_grad/Sum_1Sumgradients/pi/sub_grad/Neg-gradients/pi/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
Ś
gradients/pi/sub_grad/Reshape_1Reshapegradients/pi/sub_grad/Sum_1gradients/pi/sub_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
-gradients/pi/add_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
´
gradients/pi/add_1_grad/SumSum#gradients/pi/truediv_grad/Reshape_1-gradients/pi/add_1_grad/Sum/reduction_indices*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
h
%gradients/pi/add_1_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0

gradients/pi/add_1_grad/ReshapeReshapegradients/pi/add_1_grad/Sum%gradients/pi/add_1_grad/Reshape/shape*
_output_shapes
: *
Tshape0*
T0

-gradients/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/pi/sub_grad/Reshape_1*
_output_shapes
:*
T0*
data_formatNHWC
v
gradients/pi/Exp_1_grad/mulMul#gradients/pi/truediv_grad/Reshape_1pi/Exp_1*
_output_shapes
:*
T0
Ă
'gradients/pi/dense_2/MatMul_grad/MatMulMatMulgradients/pi/sub_grad/Reshape_1pi/dense_2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
transpose_a( *
T0
ľ
)gradients/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanhgradients/pi/sub_grad/Reshape_1*
transpose_a(*
T0*
transpose_b( *
_output_shapes
:	
Ď
gradients/AddNAddNgradients/pi/Sum_3_grad/Tilegradients/pi/mul_1_grad/Mul_1gradients/pi/Exp_1_grad/mul*
T0*
_output_shapes
:*/
_class%
#!loc:@gradients/pi/Sum_3_grad/Tile*
N
 
'gradients/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh'gradients/pi/dense_2/MatMul_grad/MatMul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
-gradients/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/pi/dense_1/Tanh_grad/TanhGrad*
data_formatNHWC*
T0*
_output_shapes	
:
Ë
'gradients/pi/dense_1/MatMul_grad/MatMulMatMul'gradients/pi/dense_1/Tanh_grad/TanhGradpi/dense_1/kernel/read*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(
ź
)gradients/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh'gradients/pi/dense_1/Tanh_grad/TanhGrad*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:


%gradients/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh'gradients/pi/dense_1/MatMul_grad/MatMul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

+gradients/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad%gradients/pi/dense/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ä
%gradients/pi/dense/MatMul_grad/MatMulMatMul%gradients/pi/dense/Tanh_grad/TanhGradpi/dense/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
transpose_b(*
T0*
transpose_a( 
ľ
'gradients/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder%gradients/pi/dense/Tanh_grad/TanhGrad*
transpose_a(*
transpose_b( *
T0*
_output_shapes
:	<
`
Reshape/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
~
ReshapeReshape'gradients/pi/dense/MatMul_grad/MatMul_1Reshape/shape*
_output_shapes	
:x*
T0*
Tshape0
b
Reshape_1/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0

	Reshape_1Reshape+gradients/pi/dense/BiasAdd_grad/BiasAddGradReshape_1/shape*
Tshape0*
T0*
_output_shapes	
:
b
Reshape_2/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙

	Reshape_2Reshape)gradients/pi/dense_1/MatMul_grad/MatMul_1Reshape_2/shape*
T0*
Tshape0*
_output_shapes

:
b
Reshape_3/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0

	Reshape_3Reshape-gradients/pi/dense_1/BiasAdd_grad/BiasAddGradReshape_3/shape*
Tshape0*
T0*
_output_shapes	
:
b
Reshape_4/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0

	Reshape_4Reshape)gradients/pi/dense_2/MatMul_grad/MatMul_1Reshape_4/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_5/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0

	Reshape_5Reshape-gradients/pi/dense_2/BiasAdd_grad/BiasAddGradReshape_5/shape*
Tshape0*
_output_shapes
:*
T0
b
Reshape_6/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
h
	Reshape_6Reshapegradients/AddNReshape_6/shape*
Tshape0*
_output_shapes
:*
T0
M
concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ś
concatConcatV2Reshape	Reshape_1	Reshape_2	Reshape_3	Reshape_4	Reshape_5	Reshape_6concat/axis*
_output_shapes

:*

Tidx0*
N*
T0
T
gradients_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
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
p
&gradients_1/pi/Mean_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:

 gradients_1/pi/Mean_grad/ReshapeReshapegradients_1/Fill&gradients_1/pi/Mean_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
f
gradients_1/pi/Mean_grad/ShapeShapepi/Sum_2*
out_type0*
T0*
_output_shapes
:
§
gradients_1/pi/Mean_grad/TileTile gradients_1/pi/Mean_grad/Reshapegradients_1/pi/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
 gradients_1/pi/Mean_grad/Shape_1Shapepi/Sum_2*
_output_shapes
:*
out_type0*
T0
c
 gradients_1/pi/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
h
gradients_1/pi/Mean_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Ľ
gradients_1/pi/Mean_grad/ProdProd gradients_1/pi/Mean_grad/Shape_1gradients_1/pi/Mean_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
j
 gradients_1/pi/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
Š
gradients_1/pi/Mean_grad/Prod_1Prod gradients_1/pi/Mean_grad/Shape_2 gradients_1/pi/Mean_grad/Const_1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
d
"gradients_1/pi/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :

 gradients_1/pi/Mean_grad/MaximumMaximumgradients_1/pi/Mean_grad/Prod_1"gradients_1/pi/Mean_grad/Maximum/y*
_output_shapes
: *
T0

!gradients_1/pi/Mean_grad/floordivFloorDivgradients_1/pi/Mean_grad/Prod gradients_1/pi/Mean_grad/Maximum*
_output_shapes
: *
T0

gradients_1/pi/Mean_grad/CastCast!gradients_1/pi/Mean_grad/floordiv*
_output_shapes
: *
Truncate( *

DstT0*

SrcT0

 gradients_1/pi/Mean_grad/truedivRealDivgradients_1/pi/Mean_grad/Tilegradients_1/pi/Mean_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
gradients_1/pi/Sum_2_grad/ShapeShapepi/sub_4*
_output_shapes
:*
T0*
out_type0

gradients_1/pi/Sum_2_grad/SizeConst*
dtype0*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
_output_shapes
: *
value	B :
ˇ
gradients_1/pi/Sum_2_grad/addAddV2pi/Sum_2/reduction_indicesgradients_1/pi/Sum_2_grad/Size*
_output_shapes
: *
T0*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape
˝
gradients_1/pi/Sum_2_grad/modFloorModgradients_1/pi/Sum_2_grad/addgradients_1/pi/Sum_2_grad/Size*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
_output_shapes
: *
T0

!gradients_1/pi/Sum_2_grad/Shape_1Const*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
dtype0*
valueB *
_output_shapes
: 

%gradients_1/pi/Sum_2_grad/range/startConst*
value	B : *2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
dtype0*
_output_shapes
: 

%gradients_1/pi/Sum_2_grad/range/deltaConst*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
value	B :*
_output_shapes
: *
dtype0
ň
gradients_1/pi/Sum_2_grad/rangeRange%gradients_1/pi/Sum_2_grad/range/startgradients_1/pi/Sum_2_grad/Size%gradients_1/pi/Sum_2_grad/range/delta*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*

Tidx0*
_output_shapes
:

$gradients_1/pi/Sum_2_grad/Fill/valueConst*
dtype0*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
_output_shapes
: *
value	B :
Ö
gradients_1/pi/Sum_2_grad/FillFill!gradients_1/pi/Sum_2_grad/Shape_1$gradients_1/pi/Sum_2_grad/Fill/value*
T0*
_output_shapes
: *

index_type0*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape

'gradients_1/pi/Sum_2_grad/DynamicStitchDynamicStitchgradients_1/pi/Sum_2_grad/rangegradients_1/pi/Sum_2_grad/modgradients_1/pi/Sum_2_grad/Shapegradients_1/pi/Sum_2_grad/Fill*
T0*
_output_shapes
:*
N*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape

#gradients_1/pi/Sum_2_grad/Maximum/yConst*
_output_shapes
: *2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
value	B :*
dtype0
Ó
!gradients_1/pi/Sum_2_grad/MaximumMaximum'gradients_1/pi/Sum_2_grad/DynamicStitch#gradients_1/pi/Sum_2_grad/Maximum/y*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
_output_shapes
:*
T0
Ë
"gradients_1/pi/Sum_2_grad/floordivFloorDivgradients_1/pi/Sum_2_grad/Shape!gradients_1/pi/Sum_2_grad/Maximum*
T0*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
_output_shapes
:
Ŕ
!gradients_1/pi/Sum_2_grad/ReshapeReshape gradients_1/pi/Mean_grad/truediv'gradients_1/pi/Sum_2_grad/DynamicStitch*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
ą
gradients_1/pi/Sum_2_grad/TileTile!gradients_1/pi/Sum_2_grad/Reshape"gradients_1/pi/Sum_2_grad/floordiv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0
g
gradients_1/pi/sub_4_grad/ShapeShapepi/add_9*
_output_shapes
:*
T0*
out_type0
p
!gradients_1/pi/sub_4_grad/Shape_1Shapepi/log_std/read*
_output_shapes
:*
out_type0*
T0
É
/gradients_1/pi/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/sub_4_grad/Shape!gradients_1/pi/sub_4_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ľ
gradients_1/pi/sub_4_grad/SumSumgradients_1/pi/Sum_2_grad/Tile/gradients_1/pi/sub_4_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Ź
!gradients_1/pi/sub_4_grad/ReshapeReshapegradients_1/pi/sub_4_grad/Sumgradients_1/pi/sub_4_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
v
gradients_1/pi/sub_4_grad/NegNeggradients_1/pi/Sum_2_grad/Tile*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
gradients_1/pi/sub_4_grad/Sum_1Sumgradients_1/pi/sub_4_grad/Neg1gradients_1/pi/sub_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ľ
#gradients_1/pi/sub_4_grad/Reshape_1Reshapegradients_1/pi/sub_4_grad/Sum_1!gradients_1/pi/sub_4_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
g
gradients_1/pi/add_9_grad/ShapeShapepi/mul_7*
out_type0*
_output_shapes
:*
T0
q
!gradients_1/pi/add_9_grad/Shape_1Shapepi/Placeholder_1*
out_type0*
T0*
_output_shapes
:
É
/gradients_1/pi/add_9_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/add_9_grad/Shape!gradients_1/pi/add_9_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
¸
gradients_1/pi/add_9_grad/SumSum!gradients_1/pi/sub_4_grad/Reshape/gradients_1/pi/add_9_grad/BroadcastGradientArgs*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
Ź
!gradients_1/pi/add_9_grad/ReshapeReshapegradients_1/pi/add_9_grad/Sumgradients_1/pi/add_9_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ź
gradients_1/pi/add_9_grad/Sum_1Sum!gradients_1/pi/sub_4_grad/Reshape1gradients_1/pi/add_9_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
˛
#gradients_1/pi/add_9_grad/Reshape_1Reshapegradients_1/pi/add_9_grad/Sum_1!gradients_1/pi/add_9_grad/Shape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
g
gradients_1/pi/mul_7_grad/ShapeShape
pi/mul_7/x*
_output_shapes
: *
T0*
out_type0
i
!gradients_1/pi/mul_7_grad/Shape_1Shapepi/sub_3*
_output_shapes
:*
out_type0*
T0
É
/gradients_1/pi/mul_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/mul_7_grad/Shape!gradients_1/pi/mul_7_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pi/mul_7_grad/MulMul!gradients_1/pi/add_9_grad/Reshapepi/sub_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients_1/pi/mul_7_grad/SumSumgradients_1/pi/mul_7_grad/Mul/gradients_1/pi/mul_7_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 

!gradients_1/pi/mul_7_grad/ReshapeReshapegradients_1/pi/mul_7_grad/Sumgradients_1/pi/mul_7_grad/Shape*
_output_shapes
: *
T0*
Tshape0

gradients_1/pi/mul_7_grad/Mul_1Mul
pi/mul_7/x!gradients_1/pi/add_9_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ş
gradients_1/pi/mul_7_grad/Sum_1Sumgradients_1/pi/mul_7_grad/Mul_11gradients_1/pi/mul_7_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
˛
#gradients_1/pi/mul_7_grad/Reshape_1Reshapegradients_1/pi/mul_7_grad/Sum_1!gradients_1/pi/mul_7_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
gradients_1/pi/sub_3_grad/ShapeShapepi/truediv_2*
T0*
_output_shapes
:*
out_type0
i
!gradients_1/pi/sub_3_grad/Shape_1Shape
pi/sub_3/y*
out_type0*
T0*
_output_shapes
: 
É
/gradients_1/pi/sub_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/sub_3_grad/Shape!gradients_1/pi/sub_3_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ş
gradients_1/pi/sub_3_grad/SumSum#gradients_1/pi/mul_7_grad/Reshape_1/gradients_1/pi/sub_3_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ź
!gradients_1/pi/sub_3_grad/ReshapeReshapegradients_1/pi/sub_3_grad/Sumgradients_1/pi/sub_3_grad/Shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
{
gradients_1/pi/sub_3_grad/NegNeg#gradients_1/pi/mul_7_grad/Reshape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
gradients_1/pi/sub_3_grad/Sum_1Sumgradients_1/pi/sub_3_grad/Neg1gradients_1/pi/sub_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ą
#gradients_1/pi/sub_3_grad/Reshape_1Reshapegradients_1/pi/sub_3_grad/Sum_1!gradients_1/pi/sub_3_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
k
#gradients_1/pi/truediv_2_grad/ShapeShapepi/add_7*
out_type0*
T0*
_output_shapes
:
m
%gradients_1/pi/truediv_2_grad/Shape_1Shapepi/add_8*
_output_shapes
:*
out_type0*
T0
Ő
3gradients_1/pi/truediv_2_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients_1/pi/truediv_2_grad/Shape%gradients_1/pi/truediv_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

%gradients_1/pi/truediv_2_grad/RealDivRealDiv!gradients_1/pi/sub_3_grad/Reshapepi/add_8*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ä
!gradients_1/pi/truediv_2_grad/SumSum%gradients_1/pi/truediv_2_grad/RealDiv3gradients_1/pi/truediv_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
¸
%gradients_1/pi/truediv_2_grad/ReshapeReshape!gradients_1/pi/truediv_2_grad/Sum#gradients_1/pi/truediv_2_grad/Shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
!gradients_1/pi/truediv_2_grad/NegNegpi/add_7*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

'gradients_1/pi/truediv_2_grad/RealDiv_1RealDiv!gradients_1/pi/truediv_2_grad/Negpi/add_8*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

'gradients_1/pi/truediv_2_grad/RealDiv_2RealDiv'gradients_1/pi/truediv_2_grad/RealDiv_1pi/add_8*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
!gradients_1/pi/truediv_2_grad/mulMul!gradients_1/pi/sub_3_grad/Reshape'gradients_1/pi/truediv_2_grad/RealDiv_2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ä
#gradients_1/pi/truediv_2_grad/Sum_1Sum!gradients_1/pi/truediv_2_grad/mul5gradients_1/pi/truediv_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ž
'gradients_1/pi/truediv_2_grad/Reshape_1Reshape#gradients_1/pi/truediv_2_grad/Sum_1%gradients_1/pi/truediv_2_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
g
gradients_1/pi/add_7_grad/ShapeShapepi/pow_2*
T0*
out_type0*
_output_shapes
:
i
!gradients_1/pi/add_7_grad/Shape_1Shapepi/Exp_3*
T0*
out_type0*
_output_shapes
:
É
/gradients_1/pi/add_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/add_7_grad/Shape!gradients_1/pi/add_7_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ź
gradients_1/pi/add_7_grad/SumSum%gradients_1/pi/truediv_2_grad/Reshape/gradients_1/pi/add_7_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ź
!gradients_1/pi/add_7_grad/ReshapeReshapegradients_1/pi/add_7_grad/Sumgradients_1/pi/add_7_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
Ŕ
gradients_1/pi/add_7_grad/Sum_1Sum%gradients_1/pi/truediv_2_grad/Reshape1gradients_1/pi/add_7_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
Ľ
#gradients_1/pi/add_7_grad/Reshape_1Reshapegradients_1/pi/add_7_grad/Sum_1!gradients_1/pi/add_7_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
g
gradients_1/pi/pow_2_grad/ShapeShapepi/sub_2*
_output_shapes
:*
T0*
out_type0
i
!gradients_1/pi/pow_2_grad/Shape_1Shape
pi/pow_2/y*
out_type0*
_output_shapes
: *
T0
É
/gradients_1/pi/pow_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/pow_2_grad/Shape!gradients_1/pi/pow_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pi/pow_2_grad/mulMul!gradients_1/pi/add_7_grad/Reshape
pi/pow_2/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
d
gradients_1/pi/pow_2_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
r
gradients_1/pi/pow_2_grad/subSub
pi/pow_2/ygradients_1/pi/pow_2_grad/sub/y*
T0*
_output_shapes
: 

gradients_1/pi/pow_2_grad/PowPowpi/sub_2gradients_1/pi/pow_2_grad/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pi/pow_2_grad/mul_1Mulgradients_1/pi/pow_2_grad/mulgradients_1/pi/pow_2_grad/Pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
gradients_1/pi/pow_2_grad/SumSumgradients_1/pi/pow_2_grad/mul_1/gradients_1/pi/pow_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ź
!gradients_1/pi/pow_2_grad/ReshapeReshapegradients_1/pi/pow_2_grad/Sumgradients_1/pi/pow_2_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
h
#gradients_1/pi/pow_2_grad/Greater/yConst*
valueB
 *    *
_output_shapes
: *
dtype0

!gradients_1/pi/pow_2_grad/GreaterGreaterpi/sub_2#gradients_1/pi/pow_2_grad/Greater/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
q
)gradients_1/pi/pow_2_grad/ones_like/ShapeShapepi/sub_2*
out_type0*
_output_shapes
:*
T0
n
)gradients_1/pi/pow_2_grad/ones_like/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
Ĺ
#gradients_1/pi/pow_2_grad/ones_likeFill)gradients_1/pi/pow_2_grad/ones_like/Shape)gradients_1/pi/pow_2_grad/ones_like/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

index_type0
Ž
 gradients_1/pi/pow_2_grad/SelectSelect!gradients_1/pi/pow_2_grad/Greaterpi/sub_2#gradients_1/pi/pow_2_grad/ones_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
x
gradients_1/pi/pow_2_grad/LogLog gradients_1/pi/pow_2_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
m
$gradients_1/pi/pow_2_grad/zeros_like	ZerosLikepi/sub_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
"gradients_1/pi/pow_2_grad/Select_1Select!gradients_1/pi/pow_2_grad/Greatergradients_1/pi/pow_2_grad/Log$gradients_1/pi/pow_2_grad/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pi/pow_2_grad/mul_2Mul!gradients_1/pi/add_7_grad/Reshapepi/pow_2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pi/pow_2_grad/mul_3Mulgradients_1/pi/pow_2_grad/mul_2"gradients_1/pi/pow_2_grad/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ş
gradients_1/pi/pow_2_grad/Sum_1Sumgradients_1/pi/pow_2_grad/mul_31gradients_1/pi/pow_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
Ą
#gradients_1/pi/pow_2_grad/Reshape_1Reshapegradients_1/pi/pow_2_grad/Sum_1!gradients_1/pi/pow_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
x
gradients_1/pi/Exp_3_grad/mulMul#gradients_1/pi/add_7_grad/Reshape_1pi/Exp_3*
_output_shapes
:*
T0
m
gradients_1/pi/sub_2_grad/ShapeShapepi/Placeholder*
_output_shapes
:*
T0*
out_type0
s
!gradients_1/pi/sub_2_grad/Shape_1Shapepi/dense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
É
/gradients_1/pi/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/sub_2_grad/Shape!gradients_1/pi/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
¸
gradients_1/pi/sub_2_grad/SumSum!gradients_1/pi/pow_2_grad/Reshape/gradients_1/pi/sub_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
Ź
!gradients_1/pi/sub_2_grad/ReshapeReshapegradients_1/pi/sub_2_grad/Sumgradients_1/pi/sub_2_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
y
gradients_1/pi/sub_2_grad/NegNeg!gradients_1/pi/pow_2_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
gradients_1/pi/sub_2_grad/Sum_1Sumgradients_1/pi/sub_2_grad/Neg1gradients_1/pi/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
˛
#gradients_1/pi/sub_2_grad/Reshape_1Reshapegradients_1/pi/sub_2_grad/Sum_1!gradients_1/pi/sub_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
gradients_1/pi/mul_5_grad/MulMulgradients_1/pi/Exp_3_grad/mulpi/log_std/read*
T0*
_output_shapes
:
y
/gradients_1/pi/mul_5_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
˛
gradients_1/pi/mul_5_grad/SumSumgradients_1/pi/mul_5_grad/Mul/gradients_1/pi/mul_5_grad/Sum/reduction_indices*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
j
'gradients_1/pi/mul_5_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0
Ł
!gradients_1/pi/mul_5_grad/ReshapeReshapegradients_1/pi/mul_5_grad/Sum'gradients_1/pi/mul_5_grad/Reshape/shape*
T0*
_output_shapes
: *
Tshape0
v
gradients_1/pi/mul_5_grad/Mul_1Mul
pi/mul_5/xgradients_1/pi/Exp_3_grad/mul*
_output_shapes
:*
T0

/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_1/pi/sub_2_grad/Reshape_1*
data_formatNHWC*
_output_shapes
:*
T0
Ä
gradients_1/AddNAddN#gradients_1/pi/sub_4_grad/Reshape_1gradients_1/pi/mul_5_grad/Mul_1*
N*6
_class,
*(loc:@gradients_1/pi/sub_4_grad/Reshape_1*
T0*
_output_shapes
:
É
)gradients_1/pi/dense_2/MatMul_grad/MatMulMatMul#gradients_1/pi/sub_2_grad/Reshape_1pi/dense_2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( *
transpose_b(
ť
+gradients_1/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh#gradients_1/pi/sub_2_grad/Reshape_1*
T0*
transpose_a(*
_output_shapes
:	*
transpose_b( 
¤
)gradients_1/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh)gradients_1/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_1/pi/dense_1/Tanh_grad/TanhGrad*
_output_shapes	
:*
data_formatNHWC*
T0
Ď
)gradients_1/pi/dense_1/MatMul_grad/MatMulMatMul)gradients_1/pi/dense_1/Tanh_grad/TanhGradpi/dense_1/kernel/read*
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(
Ŕ
+gradients_1/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh)gradients_1/pi/dense_1/Tanh_grad/TanhGrad* 
_output_shapes
:
*
transpose_a(*
T0*
transpose_b( 
 
'gradients_1/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh)gradients_1/pi/dense_1/MatMul_grad/MatMul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
-gradients_1/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_1/pi/dense/Tanh_grad/TanhGrad*
_output_shapes	
:*
T0*
data_formatNHWC
Č
'gradients_1/pi/dense/MatMul_grad/MatMulMatMul'gradients_1/pi/dense/Tanh_grad/TanhGradpi/dense/kernel/read*
transpose_b(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
T0*
transpose_a( 
š
)gradients_1/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder'gradients_1/pi/dense/Tanh_grad/TanhGrad*
_output_shapes
:	<*
T0*
transpose_a(*
transpose_b( 
b
Reshape_7/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0

	Reshape_7Reshape)gradients_1/pi/dense/MatMul_grad/MatMul_1Reshape_7/shape*
_output_shapes	
:x*
Tshape0*
T0
b
Reshape_8/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0

	Reshape_8Reshape-gradients_1/pi/dense/BiasAdd_grad/BiasAddGradReshape_8/shape*
_output_shapes	
:*
Tshape0*
T0
b
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙

	Reshape_9Reshape+gradients_1/pi/dense_1/MatMul_grad/MatMul_1Reshape_9/shape*
_output_shapes

:*
Tshape0*
T0
c
Reshape_10/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0


Reshape_10Reshape/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGradReshape_10/shape*
T0*
_output_shapes	
:*
Tshape0
c
Reshape_11/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_11Reshape+gradients_1/pi/dense_2/MatMul_grad/MatMul_1Reshape_11/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_12/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0


Reshape_12Reshape/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGradReshape_12/shape*
T0*
_output_shapes
:*
Tshape0
c
Reshape_13/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
l

Reshape_13Reshapegradients_1/AddNReshape_13/shape*
T0*
_output_shapes
:*
Tshape0
O
concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
°
concat_1ConcatV2	Reshape_7	Reshape_8	Reshape_9
Reshape_10
Reshape_11
Reshape_12
Reshape_13concat_1/axis*
T0*

Tidx0*
_output_shapes

:*
N
Z
Placeholder_9Placeholder*
dtype0*
shape:*
_output_shapes

:
L
mul_3Mulconcat_1Placeholder_9*
T0*
_output_shapes

:
Q
Const_3Const*
_output_shapes
:*
valueB: *
dtype0
X
SumSummul_3Const_3*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
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
 *  ?*
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

gradients_2/Sum_grad/ReshapeReshapegradients_2/Fill"gradients_2/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
f
gradients_2/Sum_grad/ConstConst*
valueB:*
_output_shapes
:*
dtype0

gradients_2/Sum_grad/TileTilegradients_2/Sum_grad/Reshapegradients_2/Sum_grad/Const*
T0*

Tmultiples0*
_output_shapes

:
r
gradients_2/mul_3_grad/MulMulgradients_2/Sum_grad/TilePlaceholder_9*
T0*
_output_shapes

:
o
gradients_2/mul_3_grad/Mul_1Mulgradients_2/Sum_grad/Tileconcat_1*
_output_shapes

:*
T0
`
gradients_2/concat_1_grad/RankConst*
_output_shapes
: *
value	B :*
dtype0
y
gradients_2/concat_1_grad/modFloorModconcat_1/axisgradients_2/concat_1_grad/Rank*
_output_shapes
: *
T0
j
gradients_2/concat_1_grad/ShapeConst*
_output_shapes
:*
valueB:x*
dtype0
l
!gradients_2/concat_1_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
m
!gradients_2/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:
l
!gradients_2/concat_1_grad/Shape_3Const*
_output_shapes
:*
valueB:*
dtype0
l
!gradients_2/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:
k
!gradients_2/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:
k
!gradients_2/concat_1_grad/Shape_6Const*
_output_shapes
:*
valueB:*
dtype0

&gradients_2/concat_1_grad/ConcatOffsetConcatOffsetgradients_2/concat_1_grad/modgradients_2/concat_1_grad/Shape!gradients_2/concat_1_grad/Shape_1!gradients_2/concat_1_grad/Shape_2!gradients_2/concat_1_grad/Shape_3!gradients_2/concat_1_grad/Shape_4!gradients_2/concat_1_grad/Shape_5!gradients_2/concat_1_grad/Shape_6*>
_output_shapes,
*:::::::*
N
Ŕ
gradients_2/concat_1_grad/SliceSlicegradients_2/mul_3_grad/Mul&gradients_2/concat_1_grad/ConcatOffsetgradients_2/concat_1_grad/Shape*
T0*
_output_shapes	
:x*
Index0
Ć
!gradients_2/concat_1_grad/Slice_1Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:1!gradients_2/concat_1_grad/Shape_1*
_output_shapes	
:*
T0*
Index0
Ç
!gradients_2/concat_1_grad/Slice_2Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:2!gradients_2/concat_1_grad/Shape_2*
Index0*
T0*
_output_shapes

:
Ć
!gradients_2/concat_1_grad/Slice_3Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:3!gradients_2/concat_1_grad/Shape_3*
Index0*
_output_shapes	
:*
T0
Ć
!gradients_2/concat_1_grad/Slice_4Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:4!gradients_2/concat_1_grad/Shape_4*
_output_shapes	
:*
Index0*
T0
Ĺ
!gradients_2/concat_1_grad/Slice_5Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:5!gradients_2/concat_1_grad/Shape_5*
Index0*
_output_shapes
:*
T0
Ĺ
!gradients_2/concat_1_grad/Slice_6Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:6!gradients_2/concat_1_grad/Shape_6*
Index0*
T0*
_output_shapes
:
q
 gradients_2/Reshape_7_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"<      
¨
"gradients_2/Reshape_7_grad/ReshapeReshapegradients_2/concat_1_grad/Slice gradients_2/Reshape_7_grad/Shape*
T0*
Tshape0*
_output_shapes
:	<
k
 gradients_2/Reshape_8_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0
Ś
"gradients_2/Reshape_8_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_1 gradients_2/Reshape_8_grad/Shape*
T0*
Tshape0*
_output_shapes	
:
q
 gradients_2/Reshape_9_grad/ShapeConst*
valueB"      *
_output_shapes
:*
dtype0
Ť
"gradients_2/Reshape_9_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_2 gradients_2/Reshape_9_grad/Shape*
T0*
Tshape0* 
_output_shapes
:

l
!gradients_2/Reshape_10_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0
¨
#gradients_2/Reshape_10_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_3!gradients_2/Reshape_10_grad/Shape*
Tshape0*
_output_shapes	
:*
T0
r
!gradients_2/Reshape_11_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ź
#gradients_2/Reshape_11_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_4!gradients_2/Reshape_11_grad/Shape*
_output_shapes
:	*
T0*
Tshape0
k
!gradients_2/Reshape_12_grad/ShapeConst*
valueB:*
_output_shapes
:*
dtype0
§
#gradients_2/Reshape_12_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_5!gradients_2/Reshape_12_grad/Shape*
_output_shapes
:*
T0*
Tshape0
k
!gradients_2/Reshape_13_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0
§
#gradients_2/Reshape_13_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_6!gradients_2/Reshape_13_grad/Shape*
_output_shapes
:*
T0*
Tshape0
đ
Agradients_2/gradients_1/pi/dense/MatMul_grad/MatMul_1_grad/MatMulMatMul'gradients_1/pi/dense/Tanh_grad/TanhGrad"gradients_2/Reshape_7_grad/Reshape*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
T0*
transpose_b(
×
Cgradients_2/gradients_1/pi/dense/MatMul_grad/MatMul_1_grad/MatMul_1MatMulPlaceholder"gradients_2/Reshape_7_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
Ť
Dgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ShapeShape'gradients_1/pi/dense/Tanh_grad/TanhGrad*
out_type0*
T0*
_output_shapes
:

Fgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:

Rgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
§
Tgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Const*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:

Tgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
¸
Lgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_sliceStridedSliceDgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ShapeRgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackTgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Tgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2*

begin_mask*
T0*
new_axis_mask *
end_mask *
shrink_axis_mask *
Index0*
ellipsis_mask *
_output_shapes
:

Ngradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapeConst*
valueB:*
_output_shapes
:*
dtype0

Ngradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_like/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
§
Hgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_likeFillNgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapeNgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_like/Const*
T0*

index_type0*
_output_shapes
:

Jgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
é
Egradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concatConcatV2Hgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_likeFgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/Shape_1Jgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat/axis*
N*
_output_shapes
:*
T0*

Tidx0

Tgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0
Š
Vgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
 
Vgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ŕ
Ngradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1StridedSliceDgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ShapeTgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackVgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Vgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2*
new_axis_mask *
Index0*
shrink_axis_mask *
ellipsis_mask *
end_mask *

begin_mask*
T0*
_output_shapes
:

Pgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:

Lgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
ý
Ggradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1ConcatV2Ngradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1Pgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Lgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1/axis*
T0*

Tidx0*
_output_shapes
:*
N
ô
Fgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ReshapeReshape"gradients_2/Reshape_8_grad/ReshapeEgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat*
Tshape0*
_output_shapes
:	*
T0
Ą
Cgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/TileTileFgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ReshapeGgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1*

Tmultiples0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ő
Cgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMulMatMul)gradients_1/pi/dense_1/Tanh_grad/TanhGrad"gradients_2/Reshape_9_grad/Reshape*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ű
Egradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMul_1MatMulpi/dense/Tanh"gradients_2/Reshape_9_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
transpose_a( *
T0
Ż
Fgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ShapeShape)gradients_1/pi/dense_1/Tanh_grad/TanhGrad*
T0*
out_type0*
_output_shapes
:

Hgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:

Tgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Š
Vgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Const*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
 
Vgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Â
Ngradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_sliceStridedSliceFgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ShapeTgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackVgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Vgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2*
_output_shapes
:*
T0*

begin_mask*
shrink_axis_mask *
new_axis_mask *
ellipsis_mask *
Index0*
end_mask 

Pgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapeConst*
valueB:*
_output_shapes
:*
dtype0

Pgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_like/ConstConst*
dtype0*
value	B :*
_output_shapes
: 
­
Jgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_likeFillPgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapePgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_like/Const*
_output_shapes
:*
T0*

index_type0

Lgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ń
Ggradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concatConcatV2Jgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_likeHgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/Shape_1Lgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat/axis*
_output_shapes
:*
N*
T0*

Tidx0
 
Vgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ť
Xgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
˘
Xgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ę
Pgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1StridedSliceFgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ShapeVgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackXgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Xgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2*
end_mask *

begin_mask*
Index0*
T0*
new_axis_mask *
shrink_axis_mask *
ellipsis_mask *
_output_shapes
:

Rgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Const*
valueB:*
_output_shapes
:*
dtype0

Ngradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

Igradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1ConcatV2Pgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1Rgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Ngradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
ů
Hgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ReshapeReshape#gradients_2/Reshape_10_grad/ReshapeGgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat*
Tshape0*
_output_shapes
:	*
T0
§
Egradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/TileTileHgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ReshapeIgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1*
T0*

Tmultiples0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
Cgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMulMatMul#gradients_1/pi/sub_2_grad/Reshape_1#gradients_2/Reshape_11_grad/Reshape*
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( 
Ý
Egradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMul_1MatMulpi/dense_1/Tanh#gradients_2/Reshape_11_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( *
transpose_b( 
Š
Fgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ShapeShape#gradients_1/pi/sub_2_grad/Reshape_1*
out_type0*
_output_shapes
:*
T0

Hgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0

Tgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
Š
Vgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Const*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
 
Vgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Â
Ngradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_sliceStridedSliceFgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ShapeTgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackVgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Vgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2*
ellipsis_mask *
end_mask *
new_axis_mask *

begin_mask*
_output_shapes
:*
T0*
Index0*
shrink_axis_mask 

Pgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapeConst*
valueB:*
_output_shapes
:*
dtype0

Pgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :
­
Jgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_likeFillPgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapePgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_like/Const*
T0*

index_type0*
_output_shapes
:

Lgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
ń
Ggradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concatConcatV2Jgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_likeHgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/Shape_1Lgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat/axis*
T0*
_output_shapes
:*
N*

Tidx0
 
Vgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: 
Ť
Xgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
˘
Xgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ę
Pgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1StridedSliceFgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ShapeVgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackXgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Xgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2*
ellipsis_mask *
Index0*

begin_mask*
shrink_axis_mask *
end_mask *
new_axis_mask *
_output_shapes
:*
T0

Rgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Const*
dtype0*
_output_shapes
:*
valueB:

Ngradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

Igradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1ConcatV2Pgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1Rgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Ngradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1/axis*
T0*

Tidx0*
N*
_output_shapes
:
ř
Hgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ReshapeReshape#gradients_2/Reshape_12_grad/ReshapeGgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat*
_output_shapes

:*
T0*
Tshape0
Ś
Egradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/TileTileHgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ReshapeIgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0
ś
gradients_2/AddNAddNCgradients_2/gradients_1/pi/dense/MatMul_grad/MatMul_1_grad/MatMul_1Cgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/Tile*V
_classL
JHloc:@gradients_2/gradients_1/pi/dense/MatMul_grad/MatMul_1_grad/MatMul_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
N*
T0

>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul/yConst^gradients_2/AddN*
_output_shapes
: *
valueB
 *   Ŕ*
dtype0
Č
<gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mulMulgradients_2/AddN>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul/y*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul_1Mul<gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul)gradients_1/pi/dense_1/MatMul_grad/MatMul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul_2Mul>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul_1pi/dense/Tanh*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
Agradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/TanhGradTanhGradpi/dense/Tanhgradients_2/AddN*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¤
4gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/MulMul#gradients_2/Reshape_13_grad/Reshapegradients_1/pi/Exp_3_grad/mul*
T0*
_output_shapes
:

Fgradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Sum/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
÷
4gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/SumSum4gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/MulFgradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0

>gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0
č
8gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/ReshapeReshape4gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Sum>gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Reshape/shape*
_output_shapes
: *
Tshape0*
T0

6gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Mul_1Mul
pi/mul_5/x#gradients_2/Reshape_13_grad/Reshape*
T0*
_output_shapes
:
˙
Agradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_grad/MatMulMatMulAgradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/TanhGradpi/dense_1/kernel/read*
transpose_a( *
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Cgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_grad/MatMul_1MatMulAgradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/TanhGrad)gradients_1/pi/dense_1/Tanh_grad/TanhGrad*
T0*
transpose_b( *
transpose_a(* 
_output_shapes
:

 
2gradients_2/gradients_1/pi/Exp_3_grad/mul_grad/MulMul6gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Mul_1pi/Exp_3*
_output_shapes
:*
T0
˝
4gradients_2/gradients_1/pi/Exp_3_grad/mul_grad/Mul_1Mul6gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Mul_1#gradients_1/pi/add_7_grad/Reshape_1*
_output_shapes
:*
T0

gradients_2/AddN_1AddNEgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMul_1Egradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/TileAgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*X
_classN
LJloc:@gradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMul_1*
N*
T0

@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul/yConst^gradients_2/AddN_1*
dtype0*
valueB
 *   Ŕ*
_output_shapes
: 
Î
>gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mulMulgradients_2/AddN_1@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul/y*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul_1Mul>gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul)gradients_1/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Í
@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul_2Mul@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul_1pi/dense_1/Tanh*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
Cgradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/TanhGradTanhGradpi/dense_1/Tanhgradients_2/AddN_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_2/pi/Exp_3_grad/mulMul4gradients_2/gradients_1/pi/Exp_3_grad/mul_grad/Mul_1pi/Exp_3*
_output_shapes
:*
T0

Agradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_grad/MatMulMatMulCgradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/TanhGradpi/dense_2/kernel/read*
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( 

Cgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_grad/MatMul_1MatMulCgradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/TanhGrad#gradients_1/pi/sub_2_grad/Reshape_1*
_output_shapes
:	*
transpose_b( *
transpose_a(*
T0
y
gradients_2/pi/mul_5_grad/MulMulgradients_2/pi/Exp_3_grad/mulpi/log_std/read*
_output_shapes
:*
T0
y
/gradients_2/pi/mul_5_grad/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
˛
gradients_2/pi/mul_5_grad/SumSumgradients_2/pi/mul_5_grad/Mul/gradients_2/pi/mul_5_grad/Sum/reduction_indices*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
j
'gradients_2/pi/mul_5_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Ł
!gradients_2/pi/mul_5_grad/ReshapeReshapegradients_2/pi/mul_5_grad/Sum'gradients_2/pi/mul_5_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
v
gradients_2/pi/mul_5_grad/Mul_1Mul
pi/mul_5/xgradients_2/pi/Exp_3_grad/mul*
_output_shapes
:*
T0

gradients_2/AddN_2AddNEgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMul_1Egradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/TileAgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*X
_classN
LJloc:@gradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMul_1*
N*
T0
˘
:gradients_2/gradients_1/pi/sub_2_grad/Reshape_1_grad/ShapeShapegradients_1/pi/sub_2_grad/Sum_1*
out_type0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
<gradients_2/gradients_1/pi/sub_2_grad/Reshape_1_grad/ReshapeReshapegradients_2/AddN_2:gradients_2/gradients_1/pi/sub_2_grad/Reshape_1_grad/Shape*
_output_shapes
:*
T0*
Tshape0

6gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/ShapeShapegradients_1/pi/sub_2_grad/Neg*
T0*
out_type0*
_output_shapes
:
Â
5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/SizeConst*
_output_shapes
: *
dtype0*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
value	B :
 
4gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/addAddV21gradients_1/pi/sub_2_grad/BroadcastGradientArgs:15gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Size*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
4gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/modFloorMod4gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/add5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Size*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
÷
8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape_1Shape4gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/mod*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
_output_shapes
:*
out_type0*
T0
É
<gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/range/startConst*
dtype0*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
value	B : *
_output_shapes
: 
É
<gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape
ĺ
6gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/rangeRange<gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/range/start5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Size<gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/range/delta*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
_output_shapes
:*

Tidx0
Č
;gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Fill/valueConst*
value	B :*
dtype0*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
_output_shapes
: 
ż
5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/FillFill8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape_1;gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Fill/value*
T0*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

index_type0
Ž
>gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/DynamicStitchDynamicStitch6gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/range4gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/mod6gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Fill*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
T0*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
:gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Maximum/yConst*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
_output_shapes
: *
value	B :*
dtype0
¸
8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/MaximumMaximum>gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/DynamicStitch:gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Maximum/y*
T0*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
9gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/floordivFloorDiv6gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Maximum*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
T0*
_output_shapes
:
ň
8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/ReshapeReshape<gradients_2/gradients_1/pi/sub_2_grad/Reshape_1_grad/Reshape>gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/DynamicStitch*
Tshape0*
T0*
_output_shapes
:
ö
5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/TileTile8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Reshape9gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/floordiv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tmultiples0
˘
2gradients_2/gradients_1/pi/sub_2_grad/Neg_grad/NegNeg5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

8gradients_2/gradients_1/pi/pow_2_grad/Reshape_grad/ShapeShapegradients_1/pi/pow_2_grad/Sum*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
:gradients_2/gradients_1/pi/pow_2_grad/Reshape_grad/ReshapeReshape2gradients_2/gradients_1/pi/sub_2_grad/Neg_grad/Neg8gradients_2/gradients_1/pi/pow_2_grad/Reshape_grad/Shape*
_output_shapes
:*
T0*
Tshape0

4gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/ShapeShapegradients_1/pi/pow_2_grad/mul_1*
out_type0*
_output_shapes
:*
T0
ž
3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/SizeConst*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
_output_shapes
: *
dtype0*
value	B :

2gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/addAddV2/gradients_1/pi/pow_2_grad/BroadcastGradientArgs3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Size*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape

2gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/modFloorMod2gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/add3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Size*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ń
6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape_1Shape2gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/mod*
_output_shapes
:*
T0*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
out_type0
Ĺ
:gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/range/startConst*
_output_shapes
: *
value	B : *
dtype0*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape
Ĺ
:gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/range/deltaConst*
value	B :*
dtype0*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
_output_shapes
: 
Ű
4gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/rangeRange:gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/range/start3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Size:gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/range/delta*

Tidx0*
_output_shapes
:*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape
Ä
9gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Fill/valueConst*
_output_shapes
: *
value	B :*
dtype0*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape
ˇ
3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/FillFill6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape_19gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Fill/value*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
T0*

index_type0
˘
<gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/DynamicStitchDynamicStitch4gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/range2gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/mod4gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Fill*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
N*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
T0
Ă
8gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Maximum/yConst*
_output_shapes
: *
dtype0*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
value	B :
°
6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/MaximumMaximum<gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/DynamicStitch8gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Maximum/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape

7gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/floordivFloorDiv4gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Maximum*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
T0*
_output_shapes
:
ě
6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/ReshapeReshape:gradients_2/gradients_1/pi/pow_2_grad/Reshape_grad/Reshape<gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/DynamicStitch*
_output_shapes
:*
Tshape0*
T0
đ
3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/TileTile6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Reshape7gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/floordiv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tmultiples0

6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/ShapeShapegradients_1/pi/pow_2_grad/mul*
_output_shapes
:*
out_type0*
T0

8gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Shape_1Shapegradients_1/pi/pow_2_grad/Pow*
_output_shapes
:*
out_type0*
T0

Fgradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Shape8gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Á
4gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/MulMul3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Tilegradients_1/pi/pow_2_grad/Pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
4gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/SumSum4gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/MulFgradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
ń
8gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/ReshapeReshape4gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Sum6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ă
6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Mul_1Mulgradients_1/pi/pow_2_grad/mul3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Sum_1Sum6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Mul_1Hgradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
÷
:gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Reshape_1Reshape6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Sum_18gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
|
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ShapeShapepi/sub_2*
_output_shapes
:*
out_type0*
T0

6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Shape_1Shapegradients_1/pi/pow_2_grad/sub*
out_type0*
T0*
_output_shapes
: 

Dgradients_2/gradients_1/pi/pow_2_grad/Pow_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Shape6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ć
2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mulMul:gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Reshape_1gradients_1/pi/pow_2_grad/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
y
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/sub/yConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
Ż
2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/subSubgradients_1/pi/pow_2_grad/sub4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/sub/y*
T0*
_output_shapes
: 
Š
2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/PowPowpi/sub_22gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ő
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_1Mul2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ő
2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/SumSum4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_1Dgradients_2/gradients_1/pi/pow_2_grad/Pow_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ë
6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ReshapeReshape2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Sum4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
}
8gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ˇ
6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/GreaterGreaterpi/sub_28gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Greater/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_like/ShapeShapepi/sub_2*
_output_shapes
:*
T0*
out_type0

>gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?

8gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_likeFill>gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_like/Shape>gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_like/Const*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
í
5gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/SelectSelect6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Greaterpi/sub_28gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˘
2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/LogLog5gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Select*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/zeros_like	ZerosLikepi/sub_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

7gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Select_1Select6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Greater2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Log9gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Č
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_2Mul:gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Reshape_1gradients_1/pi/pow_2_grad/Pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_3Mul4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_27gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Select_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Sum_1Sum4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_3Fgradients_2/gradients_1/pi/pow_2_grad/Pow_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
ŕ
8gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Reshape_1Reshape4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Sum_16gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
m
gradients_2/pi/sub_2_grad/ShapeShapepi/Placeholder*
T0*
_output_shapes
:*
out_type0
s
!gradients_2/pi/sub_2_grad/Shape_1Shapepi/dense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
É
/gradients_2/pi/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/pi/sub_2_grad/Shape!gradients_2/pi/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Í
gradients_2/pi/sub_2_grad/SumSum6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Reshape/gradients_2/pi/sub_2_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
Ź
!gradients_2/pi/sub_2_grad/ReshapeReshapegradients_2/pi/sub_2_grad/Sumgradients_2/pi/sub_2_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0

gradients_2/pi/sub_2_grad/NegNeg6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
gradients_2/pi/sub_2_grad/Sum_1Sumgradients_2/pi/sub_2_grad/Neg1gradients_2/pi/sub_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
˛
#gradients_2/pi/sub_2_grad/Reshape_1Reshapegradients_2/pi/sub_2_grad/Sum_1!gradients_2/pi/sub_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients_2/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_2/pi/sub_2_grad/Reshape_1*
data_formatNHWC*
T0*
_output_shapes
:
É
)gradients_2/pi/dense_2/MatMul_grad/MatMulMatMul#gradients_2/pi/sub_2_grad/Reshape_1pi/dense_2/kernel/read*
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(
ť
+gradients_2/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh#gradients_2/pi/sub_2_grad/Reshape_1*
_output_shapes
:	*
T0*
transpose_a(*
transpose_b( 
ŕ
gradients_2/AddN_3AddNCgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMul@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul_2)gradients_2/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
N*V
_classL
JHloc:@gradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMul

)gradients_2/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanhgradients_2/AddN_3*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_2/AddN_4AddNCgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_grad/MatMul_1+gradients_2/pi/dense_2/MatMul_grad/MatMul_1*V
_classL
JHloc:@gradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_grad/MatMul_1*
N*
_output_shapes
:	*
T0
Ś
/gradients_2/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_2/pi/dense_1/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:*
T0
Ď
)gradients_2/pi/dense_1/MatMul_grad/MatMulMatMul)gradients_2/pi/dense_1/Tanh_grad/TanhGradpi/dense_1/kernel/read*
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(
Ŕ
+gradients_2/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh)gradients_2/pi/dense_1/Tanh_grad/TanhGrad*
T0*
transpose_b( *
transpose_a(* 
_output_shapes
:

Ţ
gradients_2/AddN_5AddNCgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMul>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul_2)gradients_2/pi/dense_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*V
_classL
JHloc:@gradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMul*
N*
T0

'gradients_2/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanhgradients_2/AddN_5*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_2/AddN_6AddNCgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_grad/MatMul_1+gradients_2/pi/dense_1/MatMul_grad/MatMul_1*
T0*
N* 
_output_shapes
:
*V
_classL
JHloc:@gradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_grad/MatMul_1
˘
-gradients_2/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_2/pi/dense/Tanh_grad/TanhGrad*
_output_shapes	
:*
data_formatNHWC*
T0
Č
'gradients_2/pi/dense/MatMul_grad/MatMulMatMul'gradients_2/pi/dense/Tanh_grad/TanhGradpi/dense/kernel/read*
transpose_b(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
transpose_a( *
T0
š
)gradients_2/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder'gradients_2/pi/dense/Tanh_grad/TanhGrad*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	<
c
Reshape_14/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0


Reshape_14Reshape)gradients_2/pi/dense/MatMul_grad/MatMul_1Reshape_14/shape*
_output_shapes	
:x*
Tshape0*
T0
c
Reshape_15/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_15Reshape-gradients_2/pi/dense/BiasAdd_grad/BiasAddGradReshape_15/shape*
Tshape0*
T0*
_output_shapes	
:
c
Reshape_16/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
p

Reshape_16Reshapegradients_2/AddN_6Reshape_16/shape*
_output_shapes

:*
Tshape0*
T0
c
Reshape_17/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_17Reshape/gradients_2/pi/dense_1/BiasAdd_grad/BiasAddGradReshape_17/shape*
T0*
_output_shapes	
:*
Tshape0
c
Reshape_18/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
o

Reshape_18Reshapegradients_2/AddN_4Reshape_18/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_19/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0


Reshape_19Reshape/gradients_2/pi/dense_2/BiasAdd_grad/BiasAddGradReshape_19/shape*
Tshape0*
T0*
_output_shapes
:
c
Reshape_20/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
{

Reshape_20Reshapegradients_2/pi/mul_5_grad/Mul_1Reshape_20/shape*
T0*
Tshape0*
_output_shapes
:
O
concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ł
concat_2ConcatV2
Reshape_14
Reshape_15
Reshape_16
Reshape_17
Reshape_18
Reshape_19
Reshape_20concat_2/axis*
N*
_output_shapes

:*
T0*

Tidx0
L
mul_4/xConst*
valueB
 *ÍĚĚ=*
_output_shapes
: *
dtype0
K
mul_4Mulmul_4/xPlaceholder_9*
_output_shapes

:*
T0
F
add_1AddV2concat_2mul_4*
T0*
_output_shapes

:
T
gradients_3/ShapeConst*
valueB *
_output_shapes
: *
dtype0
Z
gradients_3/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
u
gradients_3/FillFillgradients_3/Shapegradients_3/grad_ys_0*
T0*
_output_shapes
: *

index_type0
o
%gradients_3/Mean_2_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:

gradients_3/Mean_2_grad/ReshapeReshapegradients_3/Fill%gradients_3/Mean_2_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
b
gradients_3/Mean_2_grad/ShapeShapemul_1*
T0*
out_type0*
_output_shapes
:
¤
gradients_3/Mean_2_grad/TileTilegradients_3/Mean_2_grad/Reshapegradients_3/Mean_2_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0
d
gradients_3/Mean_2_grad/Shape_1Shapemul_1*
T0*
out_type0*
_output_shapes
:
b
gradients_3/Mean_2_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
g
gradients_3/Mean_2_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
˘
gradients_3/Mean_2_grad/ProdProdgradients_3/Mean_2_grad/Shape_1gradients_3/Mean_2_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
i
gradients_3/Mean_2_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Ś
gradients_3/Mean_2_grad/Prod_1Prodgradients_3/Mean_2_grad/Shape_2gradients_3/Mean_2_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
c
!gradients_3/Mean_2_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 

gradients_3/Mean_2_grad/MaximumMaximumgradients_3/Mean_2_grad/Prod_1!gradients_3/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_3/Mean_2_grad/floordivFloorDivgradients_3/Mean_2_grad/Prodgradients_3/Mean_2_grad/Maximum*
_output_shapes
: *
T0

gradients_3/Mean_2_grad/CastCast gradients_3/Mean_2_grad/floordiv*
_output_shapes
: *

SrcT0*
Truncate( *

DstT0

gradients_3/Mean_2_grad/truedivRealDivgradients_3/Mean_2_grad/Tilegradients_3/Mean_2_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients_3/mul_1_grad/ShapeShapeExp*
T0*
_output_shapes
:*
out_type0
k
gradients_3/mul_1_grad/Shape_1ShapePlaceholder_3*
T0*
_output_shapes
:*
out_type0
Ŕ
,gradients_3/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/mul_1_grad/Shapegradients_3/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients_3/mul_1_grad/MulMulgradients_3/Mean_2_grad/truedivPlaceholder_3*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
gradients_3/mul_1_grad/SumSumgradients_3/mul_1_grad/Mul,gradients_3/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients_3/mul_1_grad/ReshapeReshapegradients_3/mul_1_grad/Sumgradients_3/mul_1_grad/Shape*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
w
gradients_3/mul_1_grad/Mul_1MulExpgradients_3/Mean_2_grad/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
gradients_3/mul_1_grad/Sum_1Sumgradients_3/mul_1_grad/Mul_1.gradients_3/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
Ľ
 gradients_3/mul_1_grad/Reshape_1Reshapegradients_3/mul_1_grad/Sum_1gradients_3/mul_1_grad/Shape_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
r
gradients_3/Exp_grad/mulMulgradients_3/mul_1_grad/ReshapeExp*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
`
gradients_3/sub_grad/ShapeShapepi/Sum*
_output_shapes
:*
out_type0*
T0
i
gradients_3/sub_grad/Shape_1ShapePlaceholder_6*
T0*
_output_shapes
:*
out_type0
ş
*gradients_3/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/sub_grad/Shapegradients_3/sub_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ľ
gradients_3/sub_grad/SumSumgradients_3/Exp_grad/mul*gradients_3/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 

gradients_3/sub_grad/ReshapeReshapegradients_3/sub_grad/Sumgradients_3/sub_grad/Shape*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
g
gradients_3/sub_grad/NegNeggradients_3/Exp_grad/mul*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
gradients_3/sub_grad/Sum_1Sumgradients_3/sub_grad/Neg,gradients_3/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 

gradients_3/sub_grad/Reshape_1Reshapegradients_3/sub_grad/Sum_1gradients_3/sub_grad/Shape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
e
gradients_3/pi/Sum_grad/ShapeShapepi/mul_2*
out_type0*
_output_shapes
:*
T0

gradients_3/pi/Sum_grad/SizeConst*
dtype0*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
value	B :*
_output_shapes
: 
Ż
gradients_3/pi/Sum_grad/addAddV2pi/Sum/reduction_indicesgradients_3/pi/Sum_grad/Size*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
T0*
_output_shapes
: 
ľ
gradients_3/pi/Sum_grad/modFloorModgradients_3/pi/Sum_grad/addgradients_3/pi/Sum_grad/Size*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
T0*
_output_shapes
: 

gradients_3/pi/Sum_grad/Shape_1Const*
_output_shapes
: *0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
valueB *
dtype0

#gradients_3/pi/Sum_grad/range/startConst*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
_output_shapes
: *
value	B : *
dtype0

#gradients_3/pi/Sum_grad/range/deltaConst*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
dtype0*
value	B :*
_output_shapes
: 
č
gradients_3/pi/Sum_grad/rangeRange#gradients_3/pi/Sum_grad/range/startgradients_3/pi/Sum_grad/Size#gradients_3/pi/Sum_grad/range/delta*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
_output_shapes
:*

Tidx0

"gradients_3/pi/Sum_grad/Fill/valueConst*
value	B :*
dtype0*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
_output_shapes
: 
Î
gradients_3/pi/Sum_grad/FillFillgradients_3/pi/Sum_grad/Shape_1"gradients_3/pi/Sum_grad/Fill/value*
T0*
_output_shapes
: *0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*

index_type0

%gradients_3/pi/Sum_grad/DynamicStitchDynamicStitchgradients_3/pi/Sum_grad/rangegradients_3/pi/Sum_grad/modgradients_3/pi/Sum_grad/Shapegradients_3/pi/Sum_grad/Fill*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
T0*
_output_shapes
:*
N

!gradients_3/pi/Sum_grad/Maximum/yConst*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
dtype0*
value	B :*
_output_shapes
: 
Ë
gradients_3/pi/Sum_grad/MaximumMaximum%gradients_3/pi/Sum_grad/DynamicStitch!gradients_3/pi/Sum_grad/Maximum/y*
_output_shapes
:*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
T0
Ă
 gradients_3/pi/Sum_grad/floordivFloorDivgradients_3/pi/Sum_grad/Shapegradients_3/pi/Sum_grad/Maximum*
T0*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
_output_shapes
:
¸
gradients_3/pi/Sum_grad/ReshapeReshapegradients_3/sub_grad/Reshape%gradients_3/pi/Sum_grad/DynamicStitch*
Tshape0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ť
gradients_3/pi/Sum_grad/TileTilegradients_3/pi/Sum_grad/Reshape gradients_3/pi/Sum_grad/floordiv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0
g
gradients_3/pi/mul_2_grad/ShapeShape
pi/mul_2/x*
out_type0*
T0*
_output_shapes
: 
i
!gradients_3/pi/mul_2_grad/Shape_1Shapepi/add_3*
T0*
_output_shapes
:*
out_type0
É
/gradients_3/pi/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/mul_2_grad/Shape!gradients_3/pi/mul_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
~
gradients_3/pi/mul_2_grad/MulMulgradients_3/pi/Sum_grad/Tilepi/add_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
´
gradients_3/pi/mul_2_grad/SumSumgradients_3/pi/mul_2_grad/Mul/gradients_3/pi/mul_2_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

!gradients_3/pi/mul_2_grad/ReshapeReshapegradients_3/pi/mul_2_grad/Sumgradients_3/pi/mul_2_grad/Shape*
Tshape0*
_output_shapes
: *
T0

gradients_3/pi/mul_2_grad/Mul_1Mul
pi/mul_2/xgradients_3/pi/Sum_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
gradients_3/pi/mul_2_grad/Sum_1Sumgradients_3/pi/mul_2_grad/Mul_11gradients_3/pi/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
˛
#gradients_3/pi/mul_2_grad/Reshape_1Reshapegradients_3/pi/mul_2_grad/Sum_1!gradients_3/pi/mul_2_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
g
gradients_3/pi/add_3_grad/ShapeShapepi/add_2*
_output_shapes
:*
T0*
out_type0
i
!gradients_3/pi/add_3_grad/Shape_1Shape
pi/add_3/y*
T0*
out_type0*
_output_shapes
: 
É
/gradients_3/pi/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/add_3_grad/Shape!gradients_3/pi/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ş
gradients_3/pi/add_3_grad/SumSum#gradients_3/pi/mul_2_grad/Reshape_1/gradients_3/pi/add_3_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ź
!gradients_3/pi/add_3_grad/ReshapeReshapegradients_3/pi/add_3_grad/Sumgradients_3/pi/add_3_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
ž
gradients_3/pi/add_3_grad/Sum_1Sum#gradients_3/pi/mul_2_grad/Reshape_11gradients_3/pi/add_3_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:*

Tidx0
Ą
#gradients_3/pi/add_3_grad/Reshape_1Reshapegradients_3/pi/add_3_grad/Sum_1!gradients_3/pi/add_3_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
e
gradients_3/pi/add_2_grad/ShapeShapepi/pow*
out_type0*
_output_shapes
:*
T0
i
!gradients_3/pi/add_2_grad/Shape_1Shapepi/mul_1*
T0*
_output_shapes
:*
out_type0
É
/gradients_3/pi/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/add_2_grad/Shape!gradients_3/pi/add_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
¸
gradients_3/pi/add_2_grad/SumSum!gradients_3/pi/add_3_grad/Reshape/gradients_3/pi/add_2_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
Ź
!gradients_3/pi/add_2_grad/ReshapeReshapegradients_3/pi/add_2_grad/Sumgradients_3/pi/add_2_grad/Shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
gradients_3/pi/add_2_grad/Sum_1Sum!gradients_3/pi/add_3_grad/Reshape1gradients_3/pi/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ľ
#gradients_3/pi/add_2_grad/Reshape_1Reshapegradients_3/pi/add_2_grad/Sum_1!gradients_3/pi/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
g
gradients_3/pi/pow_grad/ShapeShape
pi/truediv*
out_type0*
T0*
_output_shapes
:
e
gradients_3/pi/pow_grad/Shape_1Shapepi/pow/y*
_output_shapes
: *
T0*
out_type0
Ă
-gradients_3/pi/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/pow_grad/Shapegradients_3/pi/pow_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients_3/pi/pow_grad/mulMul!gradients_3/pi/add_2_grad/Reshapepi/pow/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
gradients_3/pi/pow_grad/sub/yConst*
valueB
 *  ?*
_output_shapes
: *
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
:˙˙˙˙˙˙˙˙˙*
T0

gradients_3/pi/pow_grad/mul_1Mulgradients_3/pi/pow_grad/mulgradients_3/pi/pow_grad/Pow*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
°
gradients_3/pi/pow_grad/SumSumgradients_3/pi/pow_grad/mul_1-gradients_3/pi/pow_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Ś
gradients_3/pi/pow_grad/ReshapeReshapegradients_3/pi/pow_grad/Sumgradients_3/pi/pow_grad/Shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
!gradients_3/pi/pow_grad/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients_3/pi/pow_grad/GreaterGreater
pi/truediv!gradients_3/pi/pow_grad/Greater/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
q
'gradients_3/pi/pow_grad/ones_like/ShapeShape
pi/truediv*
out_type0*
T0*
_output_shapes
:
l
'gradients_3/pi/pow_grad/ones_like/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
ż
!gradients_3/pi/pow_grad/ones_likeFill'gradients_3/pi/pow_grad/ones_like/Shape'gradients_3/pi/pow_grad/ones_like/Const*
T0*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
gradients_3/pi/pow_grad/SelectSelectgradients_3/pi/pow_grad/Greater
pi/truediv!gradients_3/pi/pow_grad/ones_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
gradients_3/pi/pow_grad/LogLoggradients_3/pi/pow_grad/Select*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
"gradients_3/pi/pow_grad/zeros_like	ZerosLike
pi/truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ž
 gradients_3/pi/pow_grad/Select_1Selectgradients_3/pi/pow_grad/Greatergradients_3/pi/pow_grad/Log"gradients_3/pi/pow_grad/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_3/pi/pow_grad/mul_2Mul!gradients_3/pi/add_2_grad/Reshapepi/pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_3/pi/pow_grad/mul_3Mulgradients_3/pi/pow_grad/mul_2 gradients_3/pi/pow_grad/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
´
gradients_3/pi/pow_grad/Sum_1Sumgradients_3/pi/pow_grad/mul_3/gradients_3/pi/pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

!gradients_3/pi/pow_grad/Reshape_1Reshapegradients_3/pi/pow_grad/Sum_1gradients_3/pi/pow_grad/Shape_1*
_output_shapes
: *
T0*
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
˛
gradients_3/pi/mul_1_grad/SumSumgradients_3/pi/mul_1_grad/Mul/gradients_3/pi/mul_1_grad/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
j
'gradients_3/pi/mul_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Ł
!gradients_3/pi/mul_1_grad/ReshapeReshapegradients_3/pi/mul_1_grad/Sum'gradients_3/pi/mul_1_grad/Reshape/shape*
Tshape0*
_output_shapes
: *
T0
|
gradients_3/pi/mul_1_grad/Mul_1Mul
pi/mul_1/x#gradients_3/pi/add_2_grad/Reshape_1*
_output_shapes
:*
T0
g
!gradients_3/pi/truediv_grad/ShapeShapepi/sub*
_output_shapes
:*
T0*
out_type0
m
#gradients_3/pi/truediv_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
Ď
1gradients_3/pi/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients_3/pi/truediv_grad/Shape#gradients_3/pi/truediv_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

#gradients_3/pi/truediv_grad/RealDivRealDivgradients_3/pi/pow_grad/Reshapepi/add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ž
gradients_3/pi/truediv_grad/SumSum#gradients_3/pi/truediv_grad/RealDiv1gradients_3/pi/truediv_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
˛
#gradients_3/pi/truediv_grad/ReshapeReshapegradients_3/pi/truediv_grad/Sum!gradients_3/pi/truediv_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
`
gradients_3/pi/truediv_grad/NegNegpi/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

%gradients_3/pi/truediv_grad/RealDiv_1RealDivgradients_3/pi/truediv_grad/Negpi/add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

%gradients_3/pi/truediv_grad/RealDiv_2RealDiv%gradients_3/pi/truediv_grad/RealDiv_1pi/add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
 
gradients_3/pi/truediv_grad/mulMulgradients_3/pi/pow_grad/Reshape%gradients_3/pi/truediv_grad/RealDiv_2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ž
!gradients_3/pi/truediv_grad/Sum_1Sumgradients_3/pi/truediv_grad/mul3gradients_3/pi/truediv_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
Ť
%gradients_3/pi/truediv_grad/Reshape_1Reshape!gradients_3/pi/truediv_grad/Sum_1#gradients_3/pi/truediv_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
j
gradients_3/pi/sub_grad/ShapeShapePlaceholder_1*
out_type0*
_output_shapes
:*
T0
q
gradients_3/pi/sub_grad/Shape_1Shapepi/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ă
-gradients_3/pi/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/sub_grad/Shapegradients_3/pi/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ś
gradients_3/pi/sub_grad/SumSum#gradients_3/pi/truediv_grad/Reshape-gradients_3/pi/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ś
gradients_3/pi/sub_grad/ReshapeReshapegradients_3/pi/sub_grad/Sumgradients_3/pi/sub_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
y
gradients_3/pi/sub_grad/NegNeg#gradients_3/pi/truediv_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
gradients_3/pi/sub_grad/Sum_1Sumgradients_3/pi/sub_grad/Neg/gradients_3/pi/sub_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
Ź
!gradients_3/pi/sub_grad/Reshape_1Reshapegradients_3/pi/sub_grad/Sum_1gradients_3/pi/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
/gradients_3/pi/add_1_grad/Sum/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
ş
gradients_3/pi/add_1_grad/SumSum%gradients_3/pi/truediv_grad/Reshape_1/gradients_3/pi/add_1_grad/Sum/reduction_indices*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
j
'gradients_3/pi/add_1_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *
valueB 
Ł
!gradients_3/pi/add_1_grad/ReshapeReshapegradients_3/pi/add_1_grad/Sum'gradients_3/pi/add_1_grad/Reshape/shape*
_output_shapes
: *
Tshape0*
T0

/gradients_3/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad!gradients_3/pi/sub_grad/Reshape_1*
_output_shapes
:*
T0*
data_formatNHWC
z
gradients_3/pi/Exp_1_grad/mulMul%gradients_3/pi/truediv_grad/Reshape_1pi/Exp_1*
T0*
_output_shapes
:
Ç
)gradients_3/pi/dense_2/MatMul_grad/MatMulMatMul!gradients_3/pi/sub_grad/Reshape_1pi/dense_2/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
T0
š
+gradients_3/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh!gradients_3/pi/sub_grad/Reshape_1*
T0*
_output_shapes
:	*
transpose_b( *
transpose_a(
ş
gradients_3/AddNAddNgradients_3/pi/mul_1_grad/Mul_1gradients_3/pi/Exp_1_grad/mul*
T0*
_output_shapes
:*2
_class(
&$loc:@gradients_3/pi/mul_1_grad/Mul_1*
N
¤
)gradients_3/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh)gradients_3/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
/gradients_3/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_3/pi/dense_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ď
)gradients_3/pi/dense_1/MatMul_grad/MatMulMatMul)gradients_3/pi/dense_1/Tanh_grad/TanhGradpi/dense_1/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
transpose_a( 
Ŕ
+gradients_3/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh)gradients_3/pi/dense_1/Tanh_grad/TanhGrad*
transpose_a(*
T0*
transpose_b( * 
_output_shapes
:

 
'gradients_3/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh)gradients_3/pi/dense_1/MatMul_grad/MatMul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
-gradients_3/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_3/pi/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:*
T0
Č
'gradients_3/pi/dense/MatMul_grad/MatMulMatMul'gradients_3/pi/dense/Tanh_grad/TanhGradpi/dense/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙<
š
)gradients_3/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder'gradients_3/pi/dense/Tanh_grad/TanhGrad*
transpose_b( *
_output_shapes
:	<*
transpose_a(*
T0
c
Reshape_21/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_21Reshape)gradients_3/pi/dense/MatMul_grad/MatMul_1Reshape_21/shape*
_output_shapes	
:x*
T0*
Tshape0
c
Reshape_22/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_22Reshape-gradients_3/pi/dense/BiasAdd_grad/BiasAddGradReshape_22/shape*
T0*
_output_shapes	
:*
Tshape0
c
Reshape_23/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0


Reshape_23Reshape+gradients_3/pi/dense_1/MatMul_grad/MatMul_1Reshape_23/shape*
T0*
_output_shapes

:*
Tshape0
c
Reshape_24/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_24Reshape/gradients_3/pi/dense_1/BiasAdd_grad/BiasAddGradReshape_24/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_25/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0


Reshape_25Reshape+gradients_3/pi/dense_2/MatMul_grad/MatMul_1Reshape_25/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_26/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:


Reshape_26Reshape/gradients_3/pi/dense_2/BiasAdd_grad/BiasAddGradReshape_26/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_27/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
l

Reshape_27Reshapegradients_3/AddNReshape_27/shape*
Tshape0*
T0*
_output_shapes
:
O
concat_3/axisConst*
dtype0*
value	B : *
_output_shapes
: 
ł
concat_3ConcatV2
Reshape_21
Reshape_22
Reshape_23
Reshape_24
Reshape_25
Reshape_26
Reshape_27concat_3/axis*
T0*

Tidx0*
N*
_output_shapes

:
c
Reshape_28/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
q

Reshape_28Reshapepi/dense/kernel/readReshape_28/shape*
_output_shapes	
:x*
Tshape0*
T0
c
Reshape_29/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
o

Reshape_29Reshapepi/dense/bias/readReshape_29/shape*
T0*
_output_shapes	
:*
Tshape0
c
Reshape_30/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
t

Reshape_30Reshapepi/dense_1/kernel/readReshape_30/shape*
_output_shapes

:*
T0*
Tshape0
c
Reshape_31/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
q

Reshape_31Reshapepi/dense_1/bias/readReshape_31/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_32/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
s

Reshape_32Reshapepi/dense_2/kernel/readReshape_32/shape*
_output_shapes	
:*
Tshape0*
T0
c
Reshape_33/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p

Reshape_33Reshapepi/dense_2/bias/readReshape_33/shape*
Tshape0*
T0*
_output_shapes
:
c
Reshape_34/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
k

Reshape_34Reshapepi/log_std/readReshape_34/shape*
_output_shapes
:*
Tshape0*
T0
O
concat_4/axisConst*
_output_shapes
: *
value	B : *
dtype0
ł
concat_4ConcatV2
Reshape_28
Reshape_29
Reshape_30
Reshape_31
Reshape_32
Reshape_33
Reshape_34concat_4/axis*

Tidx0*
_output_shapes

:*
T0*
N
l
Const_4Const*
_output_shapes
:*1
value(B&" <                    *
dtype0
Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B : 
¤
splitSplitVPlaceholder_9Const_4split/split_dim*

Tlen0*D
_output_shapes2
0:x::::::*
T0*
	num_split
a
Reshape_35/shapeConst*
dtype0*
_output_shapes
:*
valueB"<      
f

Reshape_35ReshapesplitReshape_35/shape*
Tshape0*
_output_shapes
:	<*
T0
[
Reshape_36/shapeConst*
_output_shapes
:*
dtype0*
valueB:
d

Reshape_36Reshapesplit:1Reshape_36/shape*
Tshape0*
_output_shapes	
:*
T0
a
Reshape_37/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
i

Reshape_37Reshapesplit:2Reshape_37/shape* 
_output_shapes
:
*
Tshape0*
T0
[
Reshape_38/shapeConst*
valueB:*
dtype0*
_output_shapes
:
d

Reshape_38Reshapesplit:3Reshape_38/shape*
Tshape0*
T0*
_output_shapes	
:
a
Reshape_39/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
h

Reshape_39Reshapesplit:4Reshape_39/shape*
_output_shapes
:	*
T0*
Tshape0
Z
Reshape_40/shapeConst*
dtype0*
valueB:*
_output_shapes
:
c

Reshape_40Reshapesplit:5Reshape_40/shape*
_output_shapes
:*
Tshape0*
T0
Z
Reshape_41/shapeConst*
valueB:*
_output_shapes
:*
dtype0
c

Reshape_41Reshapesplit:6Reshape_41/shape*
Tshape0*
_output_shapes
:*
T0
¤
AssignAssignpi/dense/kernel
Reshape_35*
use_locking(*
T0*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
validate_shape(

Assign_1Assignpi/dense/bias
Reshape_36*
use_locking(*
_output_shapes	
:*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0
Ť
Assign_2Assignpi/dense_1/kernel
Reshape_37*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

˘
Assign_3Assignpi/dense_1/bias
Reshape_38*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0
Ş
Assign_4Assignpi/dense_2/kernel
Reshape_39*
validate_shape(*
use_locking(*
_output_shapes
:	*
T0*$
_class
loc:@pi/dense_2/kernel
Ą
Assign_5Assignpi/dense_2/bias
Reshape_40*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
use_locking(

Assign_6Assign
pi/log_std
Reshape_41*
T0*
validate_shape(*
use_locking(*
_class
loc:@pi/log_std*
_output_shapes
:
]

group_depsNoOp^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6
U
sub_1SubPlaceholder_4
vf/Squeeze*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
F
powPowsub_1pow/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_5Const*
_output_shapes
:*
valueB: *
dtype0
Z
Mean_3MeanpowConst_5*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
U
sub_2SubPlaceholder_5
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
pow_1Powsub_2pow_1/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_6Const*
_output_shapes
:*
valueB: *
dtype0
\
Mean_4Meanpow_1Const_6*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
?
add_2AddV2Mean_3Mean_4*
T0*
_output_shapes
: 
T
gradients_4/ShapeConst*
valueB *
_output_shapes
: *
dtype0
Z
gradients_4/grad_ys_0Const*
valueB
 *  ?*
_output_shapes
: *
dtype0
u
gradients_4/FillFillgradients_4/Shapegradients_4/grad_ys_0*
T0*
_output_shapes
: *

index_type0
B
'gradients_4/add_2_grad/tuple/group_depsNoOp^gradients_4/Fill
˝
/gradients_4/add_2_grad/tuple/control_dependencyIdentitygradients_4/Fill(^gradients_4/add_2_grad/tuple/group_deps*#
_class
loc:@gradients_4/Fill*
_output_shapes
: *
T0
ż
1gradients_4/add_2_grad/tuple/control_dependency_1Identitygradients_4/Fill(^gradients_4/add_2_grad/tuple/group_deps*
_output_shapes
: *
T0*#
_class
loc:@gradients_4/Fill
o
%gradients_4/Mean_3_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
ľ
gradients_4/Mean_3_grad/ReshapeReshape/gradients_4/add_2_grad/tuple/control_dependency%gradients_4/Mean_3_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
`
gradients_4/Mean_3_grad/ShapeShapepow*
out_type0*
T0*
_output_shapes
:
¤
gradients_4/Mean_3_grad/TileTilegradients_4/Mean_3_grad/Reshapegradients_4/Mean_3_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients_4/Mean_3_grad/Shape_1Shapepow*
_output_shapes
:*
T0*
out_type0
b
gradients_4/Mean_3_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
g
gradients_4/Mean_3_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
˘
gradients_4/Mean_3_grad/ProdProdgradients_4/Mean_3_grad/Shape_1gradients_4/Mean_3_grad/Const*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
i
gradients_4/Mean_3_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
Ś
gradients_4/Mean_3_grad/Prod_1Prodgradients_4/Mean_3_grad/Shape_2gradients_4/Mean_3_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
c
!gradients_4/Mean_3_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0

gradients_4/Mean_3_grad/MaximumMaximumgradients_4/Mean_3_grad/Prod_1!gradients_4/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_4/Mean_3_grad/floordivFloorDivgradients_4/Mean_3_grad/Prodgradients_4/Mean_3_grad/Maximum*
T0*
_output_shapes
: 

gradients_4/Mean_3_grad/CastCast gradients_4/Mean_3_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0

gradients_4/Mean_3_grad/truedivRealDivgradients_4/Mean_3_grad/Tilegradients_4/Mean_3_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
%gradients_4/Mean_4_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
ˇ
gradients_4/Mean_4_grad/ReshapeReshape1gradients_4/add_2_grad/tuple/control_dependency_1%gradients_4/Mean_4_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
b
gradients_4/Mean_4_grad/ShapeShapepow_1*
out_type0*
_output_shapes
:*
T0
¤
gradients_4/Mean_4_grad/TileTilegradients_4/Mean_4_grad/Reshapegradients_4/Mean_4_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0
d
gradients_4/Mean_4_grad/Shape_1Shapepow_1*
_output_shapes
:*
out_type0*
T0
b
gradients_4/Mean_4_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
g
gradients_4/Mean_4_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
˘
gradients_4/Mean_4_grad/ProdProdgradients_4/Mean_4_grad/Shape_1gradients_4/Mean_4_grad/Const*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
i
gradients_4/Mean_4_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
Ś
gradients_4/Mean_4_grad/Prod_1Prodgradients_4/Mean_4_grad/Shape_2gradients_4/Mean_4_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
c
!gradients_4/Mean_4_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :

gradients_4/Mean_4_grad/MaximumMaximumgradients_4/Mean_4_grad/Prod_1!gradients_4/Mean_4_grad/Maximum/y*
_output_shapes
: *
T0

 gradients_4/Mean_4_grad/floordivFloorDivgradients_4/Mean_4_grad/Prodgradients_4/Mean_4_grad/Maximum*
_output_shapes
: *
T0

gradients_4/Mean_4_grad/CastCast gradients_4/Mean_4_grad/floordiv*
_output_shapes
: *

DstT0*
Truncate( *

SrcT0

gradients_4/Mean_4_grad/truedivRealDivgradients_4/Mean_4_grad/Tilegradients_4/Mean_4_grad/Cast*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
_
gradients_4/pow_grad/ShapeShapesub_1*
_output_shapes
:*
out_type0*
T0
_
gradients_4/pow_grad/Shape_1Shapepow/y*
_output_shapes
: *
T0*
out_type0
ş
*gradients_4/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/pow_grad/Shapegradients_4/pow_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
u
gradients_4/pow_grad/mulMulgradients_4/Mean_3_grad/truedivpow/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients_4/pow_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
c
gradients_4/pow_grad/subSubpow/ygradients_4/pow_grad/sub/y*
_output_shapes
: *
T0
n
gradients_4/pow_grad/PowPowsub_1gradients_4/pow_grad/sub*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_4/pow_grad/mul_1Mulgradients_4/pow_grad/mulgradients_4/pow_grad/Pow*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
§
gradients_4/pow_grad/SumSumgradients_4/pow_grad/mul_1*gradients_4/pow_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

gradients_4/pow_grad/ReshapeReshapegradients_4/pow_grad/Sumgradients_4/pow_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
c
gradients_4/pow_grad/Greater/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
|
gradients_4/pow_grad/GreaterGreatersub_1gradients_4/pow_grad/Greater/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
i
$gradients_4/pow_grad/ones_like/ShapeShapesub_1*
_output_shapes
:*
out_type0*
T0
i
$gradients_4/pow_grad/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
˛
gradients_4/pow_grad/ones_likeFill$gradients_4/pow_grad/ones_like/Shape$gradients_4/pow_grad/ones_like/Const*
T0*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_4/pow_grad/SelectSelectgradients_4/pow_grad/Greatersub_1gradients_4/pow_grad/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
gradients_4/pow_grad/LogLoggradients_4/pow_grad/Select*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
gradients_4/pow_grad/zeros_like	ZerosLikesub_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
gradients_4/pow_grad/Select_1Selectgradients_4/pow_grad/Greatergradients_4/pow_grad/Loggradients_4/pow_grad/zeros_like*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
u
gradients_4/pow_grad/mul_2Mulgradients_4/Mean_3_grad/truedivpow*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_4/pow_grad/mul_3Mulgradients_4/pow_grad/mul_2gradients_4/pow_grad/Select_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
gradients_4/pow_grad/Sum_1Sumgradients_4/pow_grad/mul_3,gradients_4/pow_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 

gradients_4/pow_grad/Reshape_1Reshapegradients_4/pow_grad/Sum_1gradients_4/pow_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
m
%gradients_4/pow_grad/tuple/group_depsNoOp^gradients_4/pow_grad/Reshape^gradients_4/pow_grad/Reshape_1
Ţ
-gradients_4/pow_grad/tuple/control_dependencyIdentitygradients_4/pow_grad/Reshape&^gradients_4/pow_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_class%
#!loc:@gradients_4/pow_grad/Reshape*
T0
×
/gradients_4/pow_grad/tuple/control_dependency_1Identitygradients_4/pow_grad/Reshape_1&^gradients_4/pow_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_4/pow_grad/Reshape_1*
_output_shapes
: 
a
gradients_4/pow_1_grad/ShapeShapesub_2*
T0*
out_type0*
_output_shapes
:
c
gradients_4/pow_1_grad/Shape_1Shapepow_1/y*
T0*
_output_shapes
: *
out_type0
Ŕ
,gradients_4/pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/pow_1_grad/Shapegradients_4/pow_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
y
gradients_4/pow_1_grad/mulMulgradients_4/Mean_4_grad/truedivpow_1/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
a
gradients_4/pow_1_grad/sub/yConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
i
gradients_4/pow_1_grad/subSubpow_1/ygradients_4/pow_1_grad/sub/y*
_output_shapes
: *
T0
r
gradients_4/pow_1_grad/PowPowsub_2gradients_4/pow_1_grad/sub*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_4/pow_1_grad/mul_1Mulgradients_4/pow_1_grad/mulgradients_4/pow_1_grad/Pow*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
­
gradients_4/pow_1_grad/SumSumgradients_4/pow_1_grad/mul_1,gradients_4/pow_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:

gradients_4/pow_1_grad/ReshapeReshapegradients_4/pow_1_grad/Sumgradients_4/pow_1_grad/Shape*
Tshape0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
 gradients_4/pow_1_grad/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients_4/pow_1_grad/GreaterGreatersub_2 gradients_4/pow_1_grad/Greater/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
k
&gradients_4/pow_1_grad/ones_like/ShapeShapesub_2*
out_type0*
T0*
_output_shapes
:
k
&gradients_4/pow_1_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
¸
 gradients_4/pow_1_grad/ones_likeFill&gradients_4/pow_1_grad/ones_like/Shape&gradients_4/pow_1_grad/ones_like/Const*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_4/pow_1_grad/SelectSelectgradients_4/pow_1_grad/Greatersub_2 gradients_4/pow_1_grad/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
gradients_4/pow_1_grad/LogLoggradients_4/pow_1_grad/Select*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
c
!gradients_4/pow_1_grad/zeros_like	ZerosLikesub_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ś
gradients_4/pow_1_grad/Select_1Selectgradients_4/pow_1_grad/Greatergradients_4/pow_1_grad/Log!gradients_4/pow_1_grad/zeros_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
gradients_4/pow_1_grad/mul_2Mulgradients_4/Mean_4_grad/truedivpow_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_4/pow_1_grad/mul_3Mulgradients_4/pow_1_grad/mul_2gradients_4/pow_1_grad/Select_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ą
gradients_4/pow_1_grad/Sum_1Sumgradients_4/pow_1_grad/mul_3.gradients_4/pow_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0

 gradients_4/pow_1_grad/Reshape_1Reshapegradients_4/pow_1_grad/Sum_1gradients_4/pow_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_4/pow_1_grad/tuple/group_depsNoOp^gradients_4/pow_1_grad/Reshape!^gradients_4/pow_1_grad/Reshape_1
ć
/gradients_4/pow_1_grad/tuple/control_dependencyIdentitygradients_4/pow_1_grad/Reshape(^gradients_4/pow_1_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*1
_class'
%#loc:@gradients_4/pow_1_grad/Reshape
ß
1gradients_4/pow_1_grad/tuple/control_dependency_1Identity gradients_4/pow_1_grad/Reshape_1(^gradients_4/pow_1_grad/tuple/group_deps*
_output_shapes
: *3
_class)
'%loc:@gradients_4/pow_1_grad/Reshape_1*
T0
i
gradients_4/sub_1_grad/ShapeShapePlaceholder_4*
T0*
out_type0*
_output_shapes
:
h
gradients_4/sub_1_grad/Shape_1Shape
vf/Squeeze*
T0*
out_type0*
_output_shapes
:
Ŕ
,gradients_4/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/sub_1_grad/Shapegradients_4/sub_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ž
gradients_4/sub_1_grad/SumSum-gradients_4/pow_grad/tuple/control_dependency,gradients_4/sub_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:

gradients_4/sub_1_grad/ReshapeReshapegradients_4/sub_1_grad/Sumgradients_4/sub_1_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
~
gradients_4/sub_1_grad/NegNeg-gradients_4/pow_grad/tuple/control_dependency*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ż
gradients_4/sub_1_grad/Sum_1Sumgradients_4/sub_1_grad/Neg.gradients_4/sub_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
Ľ
 gradients_4/sub_1_grad/Reshape_1Reshapegradients_4/sub_1_grad/Sum_1gradients_4/sub_1_grad/Shape_1*
Tshape0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_4/sub_1_grad/tuple/group_depsNoOp^gradients_4/sub_1_grad/Reshape!^gradients_4/sub_1_grad/Reshape_1
ć
/gradients_4/sub_1_grad/tuple/control_dependencyIdentitygradients_4/sub_1_grad/Reshape(^gradients_4/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_4/sub_1_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
1gradients_4/sub_1_grad/tuple/control_dependency_1Identity gradients_4/sub_1_grad/Reshape_1(^gradients_4/sub_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_4/sub_1_grad/Reshape_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
gradients_4/sub_2_grad/ShapeShapePlaceholder_5*
out_type0*
T0*
_output_shapes
:
h
gradients_4/sub_2_grad/Shape_1Shape
vc/Squeeze*
T0*
out_type0*
_output_shapes
:
Ŕ
,gradients_4/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/sub_2_grad/Shapegradients_4/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ŕ
gradients_4/sub_2_grad/SumSum/gradients_4/pow_1_grad/tuple/control_dependency,gradients_4/sub_2_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:

gradients_4/sub_2_grad/ReshapeReshapegradients_4/sub_2_grad/Sumgradients_4/sub_2_grad/Shape*
Tshape0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_4/sub_2_grad/NegNeg/gradients_4/pow_1_grad/tuple/control_dependency*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ż
gradients_4/sub_2_grad/Sum_1Sumgradients_4/sub_2_grad/Neg.gradients_4/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
Ľ
 gradients_4/sub_2_grad/Reshape_1Reshapegradients_4/sub_2_grad/Sum_1gradients_4/sub_2_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_4/sub_2_grad/tuple/group_depsNoOp^gradients_4/sub_2_grad/Reshape!^gradients_4/sub_2_grad/Reshape_1
ć
/gradients_4/sub_2_grad/tuple/control_dependencyIdentitygradients_4/sub_2_grad/Reshape(^gradients_4/sub_2_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*1
_class'
%#loc:@gradients_4/sub_2_grad/Reshape*
T0
ě
1gradients_4/sub_2_grad/tuple/control_dependency_1Identity gradients_4/sub_2_grad/Reshape_1(^gradients_4/sub_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_4/sub_2_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
!gradients_4/vf/Squeeze_grad/ShapeShapevf/dense_2/BiasAdd*
_output_shapes
:*
out_type0*
T0
Ä
#gradients_4/vf/Squeeze_grad/ReshapeReshape1gradients_4/sub_1_grad/tuple/control_dependency_1!gradients_4/vf/Squeeze_grad/Shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
!gradients_4/vc/Squeeze_grad/ShapeShapevc/dense_2/BiasAdd*
out_type0*
T0*
_output_shapes
:
Ä
#gradients_4/vc/Squeeze_grad/ReshapeReshape1gradients_4/sub_2_grad/tuple/control_dependency_1!gradients_4/vc/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients_4/vf/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_4/vf/Squeeze_grad/Reshape*
data_formatNHWC*
T0*
_output_shapes
:

4gradients_4/vf/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_4/vf/Squeeze_grad/Reshape0^gradients_4/vf/dense_2/BiasAdd_grad/BiasAddGrad

<gradients_4/vf/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_4/vf/Squeeze_grad/Reshape5^gradients_4/vf/dense_2/BiasAdd_grad/tuple/group_deps*6
_class,
*(loc:@gradients_4/vf/Squeeze_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_4/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_4/vf/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_4/vf/dense_2/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@gradients_4/vf/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0

/gradients_4/vc/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_4/vc/Squeeze_grad/Reshape*
T0*
_output_shapes
:*
data_formatNHWC

4gradients_4/vc/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_4/vc/Squeeze_grad/Reshape0^gradients_4/vc/dense_2/BiasAdd_grad/BiasAddGrad

<gradients_4/vc/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_4/vc/Squeeze_grad/Reshape5^gradients_4/vc/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*6
_class,
*(loc:@gradients_4/vc/Squeeze_grad/Reshape*
T0

>gradients_4/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_4/vc/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_4/vc/dense_2/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@gradients_4/vc/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
â
)gradients_4/vf/dense_2/MatMul_grad/MatMulMatMul<gradients_4/vf/dense_2/BiasAdd_grad/tuple/control_dependencyvf/dense_2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Ô
+gradients_4/vf/dense_2/MatMul_grad/MatMul_1MatMulvf/dense_1/Tanh<gradients_4/vf/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	*
T0*
transpose_b( 

3gradients_4/vf/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_4/vf/dense_2/MatMul_grad/MatMul,^gradients_4/vf/dense_2/MatMul_grad/MatMul_1

;gradients_4/vf/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_4/vf/dense_2/MatMul_grad/MatMul4^gradients_4/vf/dense_2/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients_4/vf/dense_2/MatMul_grad/MatMul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_4/vf/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_4/vf/dense_2/MatMul_grad/MatMul_14^gradients_4/vf/dense_2/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	*>
_class4
20loc:@gradients_4/vf/dense_2/MatMul_grad/MatMul_1
â
)gradients_4/vc/dense_2/MatMul_grad/MatMulMatMul<gradients_4/vc/dense_2/BiasAdd_grad/tuple/control_dependencyvc/dense_2/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_b(
Ô
+gradients_4/vc/dense_2/MatMul_grad/MatMul_1MatMulvc/dense_1/Tanh<gradients_4/vc/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	*
T0*
transpose_b( 

3gradients_4/vc/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_4/vc/dense_2/MatMul_grad/MatMul,^gradients_4/vc/dense_2/MatMul_grad/MatMul_1

;gradients_4/vc/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_4/vc/dense_2/MatMul_grad/MatMul4^gradients_4/vc/dense_2/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*<
_class2
0.loc:@gradients_4/vc/dense_2/MatMul_grad/MatMul

=gradients_4/vc/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_4/vc/dense_2/MatMul_grad/MatMul_14^gradients_4/vc/dense_2/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	*>
_class4
20loc:@gradients_4/vc/dense_2/MatMul_grad/MatMul_1
ś
)gradients_4/vf/dense_1/Tanh_grad/TanhGradTanhGradvf/dense_1/Tanh;gradients_4/vf/dense_2/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ś
)gradients_4/vc/dense_1/Tanh_grad/TanhGradTanhGradvc/dense_1/Tanh;gradients_4/vc/dense_2/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
/gradients_4/vf/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_4/vf/dense_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:

4gradients_4/vf/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_4/vf/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_4/vf/dense_1/Tanh_grad/TanhGrad

<gradients_4/vf/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_4/vf/dense_1/Tanh_grad/TanhGrad5^gradients_4/vf/dense_1/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_4/vf/dense_1/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_4/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_4/vf/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_4/vf/dense_1/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_4/vf/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ś
/gradients_4/vc/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_4/vc/dense_1/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:*
T0

4gradients_4/vc/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_4/vc/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_4/vc/dense_1/Tanh_grad/TanhGrad

<gradients_4/vc/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_4/vc/dense_1/Tanh_grad/TanhGrad5^gradients_4/vc/dense_1/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_4/vc/dense_1/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_4/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_4/vc/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_4/vc/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*B
_class8
64loc:@gradients_4/vc/dense_1/BiasAdd_grad/BiasAddGrad
â
)gradients_4/vf/dense_1/MatMul_grad/MatMulMatMul<gradients_4/vf/dense_1/BiasAdd_grad/tuple/control_dependencyvf/dense_1/kernel/read*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(
Ó
+gradients_4/vf/dense_1/MatMul_grad/MatMul_1MatMulvf/dense/Tanh<gradients_4/vf/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(* 
_output_shapes
:
*
T0*
transpose_b( 

3gradients_4/vf/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_4/vf/dense_1/MatMul_grad/MatMul,^gradients_4/vf/dense_1/MatMul_grad/MatMul_1

;gradients_4/vf/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_4/vf/dense_1/MatMul_grad/MatMul4^gradients_4/vf/dense_1/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_4/vf/dense_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_4/vf/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_4/vf/dense_1/MatMul_grad/MatMul_14^gradients_4/vf/dense_1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*>
_class4
20loc:@gradients_4/vf/dense_1/MatMul_grad/MatMul_1
â
)gradients_4/vc/dense_1/MatMul_grad/MatMulMatMul<gradients_4/vc/dense_1/BiasAdd_grad/tuple/control_dependencyvc/dense_1/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ó
+gradients_4/vc/dense_1/MatMul_grad/MatMul_1MatMulvc/dense/Tanh<gradients_4/vc/dense_1/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
T0*
transpose_a(*
transpose_b( 

3gradients_4/vc/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_4/vc/dense_1/MatMul_grad/MatMul,^gradients_4/vc/dense_1/MatMul_grad/MatMul_1

;gradients_4/vc/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_4/vc/dense_1/MatMul_grad/MatMul4^gradients_4/vc/dense_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*<
_class2
0.loc:@gradients_4/vc/dense_1/MatMul_grad/MatMul*
T0

=gradients_4/vc/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_4/vc/dense_1/MatMul_grad/MatMul_14^gradients_4/vc/dense_1/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
*>
_class4
20loc:@gradients_4/vc/dense_1/MatMul_grad/MatMul_1
˛
'gradients_4/vf/dense/Tanh_grad/TanhGradTanhGradvf/dense/Tanh;gradients_4/vf/dense_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˛
'gradients_4/vc/dense/Tanh_grad/TanhGradTanhGradvc/dense/Tanh;gradients_4/vc/dense_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˘
-gradients_4/vf/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_4/vf/dense/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:

2gradients_4/vf/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_4/vf/dense/BiasAdd_grad/BiasAddGrad(^gradients_4/vf/dense/Tanh_grad/TanhGrad

:gradients_4/vf/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_4/vf/dense/Tanh_grad/TanhGrad3^gradients_4/vf/dense/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_4/vf/dense/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_4/vf/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_4/vf/dense/BiasAdd_grad/BiasAddGrad3^gradients_4/vf/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*@
_class6
42loc:@gradients_4/vf/dense/BiasAdd_grad/BiasAddGrad
˘
-gradients_4/vc/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_4/vc/dense/Tanh_grad/TanhGrad*
_output_shapes	
:*
T0*
data_formatNHWC

2gradients_4/vc/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_4/vc/dense/BiasAdd_grad/BiasAddGrad(^gradients_4/vc/dense/Tanh_grad/TanhGrad

:gradients_4/vc/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_4/vc/dense/Tanh_grad/TanhGrad3^gradients_4/vc/dense/BiasAdd_grad/tuple/group_deps*:
_class0
.,loc:@gradients_4/vc/dense/Tanh_grad/TanhGrad*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_4/vc/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_4/vc/dense/BiasAdd_grad/BiasAddGrad3^gradients_4/vc/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*@
_class6
42loc:@gradients_4/vc/dense/BiasAdd_grad/BiasAddGrad
Ű
'gradients_4/vf/dense/MatMul_grad/MatMulMatMul:gradients_4/vf/dense/BiasAdd_grad/tuple/control_dependencyvf/dense/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
transpose_a( *
transpose_b(*
T0
Ě
)gradients_4/vf/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_4/vf/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes
:	<

1gradients_4/vf/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_4/vf/dense/MatMul_grad/MatMul*^gradients_4/vf/dense/MatMul_grad/MatMul_1

9gradients_4/vf/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_4/vf/dense/MatMul_grad/MatMul2^gradients_4/vf/dense/MatMul_grad/tuple/group_deps*:
_class0
.,loc:@gradients_4/vf/dense/MatMul_grad/MatMul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<

;gradients_4/vf/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_4/vf/dense/MatMul_grad/MatMul_12^gradients_4/vf/dense/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_4/vf/dense/MatMul_grad/MatMul_1*
_output_shapes
:	<
Ű
'gradients_4/vc/dense/MatMul_grad/MatMulMatMul:gradients_4/vc/dense/BiasAdd_grad/tuple/control_dependencyvc/dense/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<*
transpose_b(*
T0*
transpose_a( 
Ě
)gradients_4/vc/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_4/vc/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
_output_shapes
:	<*
T0

1gradients_4/vc/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_4/vc/dense/MatMul_grad/MatMul*^gradients_4/vc/dense/MatMul_grad/MatMul_1

9gradients_4/vc/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_4/vc/dense/MatMul_grad/MatMul2^gradients_4/vc/dense/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_4/vc/dense/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<

;gradients_4/vc/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_4/vc/dense/MatMul_grad/MatMul_12^gradients_4/vc/dense/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_4/vc/dense/MatMul_grad/MatMul_1*
_output_shapes
:	<
c
Reshape_42/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_42Reshape;gradients_4/vf/dense/MatMul_grad/tuple/control_dependency_1Reshape_42/shape*
T0*
Tshape0*
_output_shapes	
:x
c
Reshape_43/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_43Reshape<gradients_4/vf/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_43/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_44/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_44Reshape=gradients_4/vf/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_44/shape*
T0*
_output_shapes

:*
Tshape0
c
Reshape_45/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_45Reshape>gradients_4/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_45/shape*
_output_shapes	
:*
T0*
Tshape0
c
Reshape_46/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_46Reshape=gradients_4/vf/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_46/shape*
_output_shapes	
:*
Tshape0*
T0
c
Reshape_47/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_47Reshape>gradients_4/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_47/shape*
_output_shapes
:*
Tshape0*
T0
c
Reshape_48/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0


Reshape_48Reshape;gradients_4/vc/dense/MatMul_grad/tuple/control_dependency_1Reshape_48/shape*
_output_shapes	
:x*
T0*
Tshape0
c
Reshape_49/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0


Reshape_49Reshape<gradients_4/vc/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_49/shape*
Tshape0*
T0*
_output_shapes	
:
c
Reshape_50/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:


Reshape_50Reshape=gradients_4/vc/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_50/shape*
Tshape0*
_output_shapes

:*
T0
c
Reshape_51/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_51Reshape>gradients_4/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_51/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_52/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_52Reshape=gradients_4/vc/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_52/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_53/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:


Reshape_53Reshape>gradients_4/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_53/shape*
_output_shapes
:*
Tshape0*
T0
O
concat_5/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ď
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
Reshape_53concat_5/axis*

Tidx0*
T0*
N*
_output_shapes

:ü	
j
PyFuncPyFuncconcat_5*
Tout
2*
token
pyfunc_0*
Tin
2*
_output_shapes

:ü	

Const_7Const*
dtype0*E
value<B:"0 <                  <                 *
_output_shapes
:
S
split_1/split_dimConst*
dtype0*
_output_shapes
: *
value	B : 
Ĺ
split_1SplitVPyFuncConst_7split_1/split_dim*

Tlen0*
T0*h
_output_shapesV
T:x::::::x:::::*
	num_split
a
Reshape_54/shapeConst*
dtype0*
valueB"<      *
_output_shapes
:
h

Reshape_54Reshapesplit_1Reshape_54/shape*
Tshape0*
T0*
_output_shapes
:	<
[
Reshape_55/shapeConst*
dtype0*
_output_shapes
:*
valueB:
f

Reshape_55Reshape	split_1:1Reshape_55/shape*
T0*
Tshape0*
_output_shapes	
:
a
Reshape_56/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
k

Reshape_56Reshape	split_1:2Reshape_56/shape*
Tshape0* 
_output_shapes
:
*
T0
[
Reshape_57/shapeConst*
_output_shapes
:*
dtype0*
valueB:
f

Reshape_57Reshape	split_1:3Reshape_57/shape*
_output_shapes	
:*
Tshape0*
T0
a
Reshape_58/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
j

Reshape_58Reshape	split_1:4Reshape_58/shape*
Tshape0*
_output_shapes
:	*
T0
Z
Reshape_59/shapeConst*
_output_shapes
:*
dtype0*
valueB:
e

Reshape_59Reshape	split_1:5Reshape_59/shape*
T0*
Tshape0*
_output_shapes
:
a
Reshape_60/shapeConst*
_output_shapes
:*
dtype0*
valueB"<      
j

Reshape_60Reshape	split_1:6Reshape_60/shape*
T0*
Tshape0*
_output_shapes
:	<
[
Reshape_61/shapeConst*
dtype0*
_output_shapes
:*
valueB:
f

Reshape_61Reshape	split_1:7Reshape_61/shape*
_output_shapes	
:*
Tshape0*
T0
a
Reshape_62/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
k

Reshape_62Reshape	split_1:8Reshape_62/shape*
Tshape0*
T0* 
_output_shapes
:

[
Reshape_63/shapeConst*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_63Reshape	split_1:9Reshape_63/shape*
_output_shapes	
:*
T0*
Tshape0
a
Reshape_64/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_64Reshape
split_1:10Reshape_64/shape*
T0*
_output_shapes
:	*
Tshape0
Z
Reshape_65/shapeConst*
valueB:*
_output_shapes
:*
dtype0
f

Reshape_65Reshape
split_1:11Reshape_65/shape*
_output_shapes
:*
T0*
Tshape0

beta1_power/initial_valueConst*
dtype0*
valueB
 *fff?*
_output_shapes
: * 
_class
loc:@vc/dense/bias

beta1_power
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: * 
_class
loc:@vc/dense/bias
°
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias
l
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias

beta2_power/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *wž?* 
_class
loc:@vc/dense/bias

beta2_power
VariableV2*
shared_name * 
_class
loc:@vc/dense/bias*
_output_shapes
: *
shape: *
	container *
dtype0
°
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(
l
beta2_power/readIdentitybeta2_power*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0
Ť
6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*"
_class
loc:@vf/dense/kernel*
valueB"<      *
_output_shapes
:

,vf/dense/kernel/Adam/Initializer/zeros/ConstConst*"
_class
loc:@vf/dense/kernel*
dtype0*
valueB
 *    *
_output_shapes
: 
ô
&vf/dense/kernel/Adam/Initializer/zerosFill6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vf/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	<*

index_type0*
T0*"
_class
loc:@vf/dense/kernel
Ž
vf/dense/kernel/Adam
VariableV2*
_output_shapes
:	<*
	container *
shared_name *
dtype0*"
_class
loc:@vf/dense/kernel*
shape:	<
Ú
vf/dense/kernel/Adam/AssignAssignvf/dense/kernel/Adam&vf/dense/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
T0

vf/dense/kernel/Adam/readIdentityvf/dense/kernel/Adam*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
T0
­
8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@vf/dense/kernel*
valueB"<      *
dtype0*
_output_shapes
:

.vf/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*"
_class
loc:@vf/dense/kernel*
_output_shapes
: 
ú
(vf/dense/kernel/Adam_1/Initializer/zerosFill8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vf/dense/kernel/Adam_1/Initializer/zeros/Const*"
_class
loc:@vf/dense/kernel*
T0*

index_type0*
_output_shapes
:	<
°
vf/dense/kernel/Adam_1
VariableV2*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
dtype0*
	container *
shape:	<*
shared_name 
ŕ
vf/dense/kernel/Adam_1/AssignAssignvf/dense/kernel/Adam_1(vf/dense/kernel/Adam_1/Initializer/zeros*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
validate_shape(*
T0*
use_locking(

vf/dense/kernel/Adam_1/readIdentityvf/dense/kernel/Adam_1*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	<

$vf/dense/bias/Adam/Initializer/zerosConst* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
valueB*    *
dtype0
˘
vf/dense/bias/Adam
VariableV2*
shared_name *
shape:*
dtype0*
	container *
_output_shapes	
:* 
_class
loc:@vf/dense/bias
Î
vf/dense/bias/Adam/AssignAssignvf/dense/bias/Adam$vf/dense/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0*
use_locking(

vf/dense/bias/Adam/readIdentityvf/dense/bias/Adam*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias

&vf/dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
¤
vf/dense/bias/Adam_1
VariableV2*
shared_name *
_output_shapes	
:*
shape:*
dtype0* 
_class
loc:@vf/dense/bias*
	container 
Ô
vf/dense/bias/Adam_1/AssignAssignvf/dense/bias/Adam_1&vf/dense/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vf/dense/bias*
use_locking(

vf/dense/bias/Adam_1/readIdentityvf/dense/bias/Adam_1*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0
Ż
8vf/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*$
_class
loc:@vf/dense_1/kernel*
valueB"      *
_output_shapes
:

.vf/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *$
_class
loc:@vf/dense_1/kernel*
valueB
 *    *
dtype0
ý
(vf/dense_1/kernel/Adam/Initializer/zerosFill8vf/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vf/dense_1/kernel/Adam/Initializer/zeros/Const*

index_type0*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:

´
vf/dense_1/kernel/Adam
VariableV2*
shared_name *$
_class
loc:@vf/dense_1/kernel*
shape:
* 
_output_shapes
:
*
	container *
dtype0
ă
vf/dense_1/kernel/Adam/AssignAssignvf/dense_1/kernel/Adam(vf/dense_1/kernel/Adam/Initializer/zeros*
T0* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(

vf/dense_1/kernel/Adam/readIdentityvf/dense_1/kernel/Adam* 
_output_shapes
:
*
T0*$
_class
loc:@vf/dense_1/kernel
ą
:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *
_output_shapes
:*$
_class
loc:@vf/dense_1/kernel*
dtype0

0vf/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*$
_class
loc:@vf/dense_1/kernel*
valueB
 *    *
_output_shapes
: 

*vf/dense_1/kernel/Adam_1/Initializer/zerosFill:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vf/dense_1/kernel/Adam_1/Initializer/zeros/Const*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:
*

index_type0
ś
vf/dense_1/kernel/Adam_1
VariableV2*
shared_name *
	container *$
_class
loc:@vf/dense_1/kernel*
shape:
*
dtype0* 
_output_shapes
:

é
vf/dense_1/kernel/Adam_1/AssignAssignvf/dense_1/kernel/Adam_1*vf/dense_1/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(

vf/dense_1/kernel/Adam_1/readIdentityvf/dense_1/kernel/Adam_1*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0

&vf/dense_1/bias/Adam/Initializer/zerosConst*
valueB*    *"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
dtype0
Ś
vf/dense_1/bias/Adam
VariableV2*
shared_name *
	container *
shape:*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
dtype0
Ö
vf/dense_1/bias/Adam/AssignAssignvf/dense_1/bias/Adam&vf/dense_1/bias/Adam/Initializer/zeros*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:*
T0

vf/dense_1/bias/Adam/readIdentityvf/dense_1/bias/Adam*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:

(vf/dense_1/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_output_shapes	
:*
dtype0*"
_class
loc:@vf/dense_1/bias
¨
vf/dense_1/bias/Adam_1
VariableV2*"
_class
loc:@vf/dense_1/bias*
shape:*
_output_shapes	
:*
	container *
shared_name *
dtype0
Ü
vf/dense_1/bias/Adam_1/AssignAssignvf/dense_1/bias/Adam_1(vf/dense_1/bias/Adam_1/Initializer/zeros*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(*
use_locking(

vf/dense_1/bias/Adam_1/readIdentityvf/dense_1/bias/Adam_1*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias
Ľ
(vf/dense_2/kernel/Adam/Initializer/zerosConst*
dtype0*$
_class
loc:@vf/dense_2/kernel*
valueB	*    *
_output_shapes
:	
˛
vf/dense_2/kernel/Adam
VariableV2*
shared_name *$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
dtype0*
	container *
shape:	
â
vf/dense_2/kernel/Adam/AssignAssignvf/dense_2/kernel/Adam(vf/dense_2/kernel/Adam/Initializer/zeros*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(

vf/dense_2/kernel/Adam/readIdentityvf/dense_2/kernel/Adam*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
T0
§
*vf/dense_2/kernel/Adam_1/Initializer/zerosConst*
valueB	*    *$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
dtype0
´
vf/dense_2/kernel/Adam_1
VariableV2*
shared_name *$
_class
loc:@vf/dense_2/kernel*
shape:	*
dtype0*
_output_shapes
:	*
	container 
č
vf/dense_2/kernel/Adam_1/AssignAssignvf/dense_2/kernel/Adam_1*vf/dense_2/kernel/Adam_1/Initializer/zeros*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	*
use_locking(

vf/dense_2/kernel/Adam_1/readIdentityvf/dense_2/kernel/Adam_1*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	

&vf/dense_2/bias/Adam/Initializer/zerosConst*
_output_shapes
:*
valueB*    *"
_class
loc:@vf/dense_2/bias*
dtype0
¤
vf/dense_2/bias/Adam
VariableV2*
shared_name *
	container *
_output_shapes
:*
shape:*"
_class
loc:@vf/dense_2/bias*
dtype0
Ő
vf/dense_2/bias/Adam/AssignAssignvf/dense_2/bias/Adam&vf/dense_2/bias/Adam/Initializer/zeros*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
validate_shape(

vf/dense_2/bias/Adam/readIdentityvf/dense_2/bias/Adam*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:

(vf/dense_2/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@vf/dense_2/bias*
valueB*    *
_output_shapes
:*
dtype0
Ś
vf/dense_2/bias/Adam_1
VariableV2*
_output_shapes
:*
dtype0*
shape:*
	container *
shared_name *"
_class
loc:@vf/dense_2/bias
Ű
vf/dense_2/bias/Adam_1/AssignAssignvf/dense_2/bias/Adam_1(vf/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:

vf/dense_2/bias/Adam_1/readIdentityvf/dense_2/bias/Adam_1*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:
Ť
6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@vc/dense/kernel*
valueB"<      *
_output_shapes
:*
dtype0

,vc/dense/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0*"
_class
loc:@vc/dense/kernel
ô
&vc/dense/kernel/Adam/Initializer/zerosFill6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vc/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	<*

index_type0*
T0*"
_class
loc:@vc/dense/kernel
Ž
vc/dense/kernel/Adam
VariableV2*
shape:	<*"
_class
loc:@vc/dense/kernel*
shared_name *
dtype0*
_output_shapes
:	<*
	container 
Ú
vc/dense/kernel/Adam/AssignAssignvc/dense/kernel/Adam&vc/dense/kernel/Adam/Initializer/zeros*
T0*
validate_shape(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
use_locking(

vc/dense/kernel/Adam/readIdentityvc/dense/kernel/Adam*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<
­
8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"<      *
_output_shapes
:*
dtype0*"
_class
loc:@vc/dense/kernel

.vc/dense/kernel/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@vc/dense/kernel*
_output_shapes
: *
valueB
 *    *
dtype0
ú
(vc/dense/kernel/Adam_1/Initializer/zerosFill8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vc/dense/kernel/Adam_1/Initializer/zeros/Const*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
T0*

index_type0
°
vc/dense/kernel/Adam_1
VariableV2*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
	container *
shared_name *
dtype0*
shape:	<
ŕ
vc/dense/kernel/Adam_1/AssignAssignvc/dense/kernel/Adam_1(vc/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<

vc/dense/kernel/Adam_1/readIdentityvc/dense/kernel/Adam_1*
_output_shapes
:	<*
T0*"
_class
loc:@vc/dense/kernel

$vc/dense/bias/Adam/Initializer/zerosConst*
valueB*    *
_output_shapes	
:*
dtype0* 
_class
loc:@vc/dense/bias
˘
vc/dense/bias/Adam
VariableV2*
dtype0*
	container *
shape:*
shared_name *
_output_shapes	
:* 
_class
loc:@vc/dense/bias
Î
vc/dense/bias/Adam/AssignAssignvc/dense/bias/Adam$vc/dense/bias/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:

vc/dense/bias/Adam/readIdentityvc/dense/bias/Adam* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0

&vc/dense/bias/Adam_1/Initializer/zerosConst* 
_class
loc:@vc/dense/bias*
dtype0*
_output_shapes	
:*
valueB*    
¤
vc/dense/bias/Adam_1
VariableV2*
shape:* 
_class
loc:@vc/dense/bias*
shared_name *
_output_shapes	
:*
	container *
dtype0
Ô
vc/dense/bias/Adam_1/AssignAssignvc/dense/bias/Adam_1&vc/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(

vc/dense/bias/Adam_1/readIdentityvc/dense/bias/Adam_1*
_output_shapes	
:*
T0* 
_class
loc:@vc/dense/bias
Ż
8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@vc/dense_1/kernel*
_output_shapes
:*
dtype0*
valueB"      

.vc/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *$
_class
loc:@vc/dense_1/kernel*
dtype0*
valueB
 *    
ý
(vc/dense_1/kernel/Adam/Initializer/zerosFill8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vc/dense_1/kernel/Adam/Initializer/zeros/Const*
T0* 
_output_shapes
:
*

index_type0*$
_class
loc:@vc/dense_1/kernel
´
vc/dense_1/kernel/Adam
VariableV2*
	container *
shared_name *
shape:
*
dtype0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

ă
vc/dense_1/kernel/Adam/AssignAssignvc/dense_1/kernel/Adam(vc/dense_1/kernel/Adam/Initializer/zeros* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(

vc/dense_1/kernel/Adam/readIdentityvc/dense_1/kernel/Adam*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

ą
:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*$
_class
loc:@vc/dense_1/kernel*
valueB"      *
dtype0
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
*vc/dense_1/kernel/Adam_1/Initializer/zerosFill:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vc/dense_1/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0*

index_type0
ś
vc/dense_1/kernel/Adam_1
VariableV2*
dtype0*
shape:
* 
_output_shapes
:
*
	container *$
_class
loc:@vc/dense_1/kernel*
shared_name 
é
vc/dense_1/kernel/Adam_1/AssignAssignvc/dense_1/kernel/Adam_1*vc/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:


vc/dense_1/kernel/Adam_1/readIdentityvc/dense_1/kernel/Adam_1* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel

&vc/dense_1/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*
dtype0*
valueB*    *"
_class
loc:@vc/dense_1/bias
Ś
vc/dense_1/bias/Adam
VariableV2*
shared_name *"
_class
loc:@vc/dense_1/bias*
shape:*
dtype0*
	container *
_output_shapes	
:
Ö
vc/dense_1/bias/Adam/AssignAssignvc/dense_1/bias/Adam&vc/dense_1/bias/Adam/Initializer/zeros*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(

vc/dense_1/bias/Adam/readIdentityvc/dense_1/bias/Adam*
_output_shapes	
:*
T0*"
_class
loc:@vc/dense_1/bias

(vc/dense_1/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@vc/dense_1/bias*
valueB*    *
_output_shapes	
:*
dtype0
¨
vc/dense_1/bias/Adam_1
VariableV2*
shape:*
_output_shapes	
:*
shared_name *
dtype0*"
_class
loc:@vc/dense_1/bias*
	container 
Ü
vc/dense_1/bias/Adam_1/AssignAssignvc/dense_1/bias/Adam_1(vc/dense_1/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:

vc/dense_1/bias/Adam_1/readIdentityvc/dense_1/bias/Adam_1*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:
Ľ
(vc/dense_2/kernel/Adam/Initializer/zerosConst*
_output_shapes
:	*
valueB	*    *$
_class
loc:@vc/dense_2/kernel*
dtype0
˛
vc/dense_2/kernel/Adam
VariableV2*
shape:	*$
_class
loc:@vc/dense_2/kernel*
	container *
_output_shapes
:	*
shared_name *
dtype0
â
vc/dense_2/kernel/Adam/AssignAssignvc/dense_2/kernel/Adam(vc/dense_2/kernel/Adam/Initializer/zeros*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	

vc/dense_2/kernel/Adam/readIdentityvc/dense_2/kernel/Adam*
T0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
§
*vc/dense_2/kernel/Adam_1/Initializer/zerosConst*
_output_shapes
:	*
valueB	*    *
dtype0*$
_class
loc:@vc/dense_2/kernel
´
vc/dense_2/kernel/Adam_1
VariableV2*
shape:	*
	container *
dtype0*
shared_name *
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
č
vc/dense_2/kernel/Adam_1/AssignAssignvc/dense_2/kernel/Adam_1*vc/dense_2/kernel/Adam_1/Initializer/zeros*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(*
T0

vc/dense_2/kernel/Adam_1/readIdentityvc/dense_2/kernel/Adam_1*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
T0

&vc/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
¤
vc/dense_2/bias/Adam
VariableV2*
shape:*"
_class
loc:@vc/dense_2/bias*
dtype0*
_output_shapes
:*
	container *
shared_name 
Ő
vc/dense_2/bias/Adam/AssignAssignvc/dense_2/bias/Adam&vc/dense_2/bias/Adam/Initializer/zeros*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(*
T0

vc/dense_2/bias/Adam/readIdentityvc/dense_2/bias/Adam*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:

(vc/dense_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
dtype0*"
_class
loc:@vc/dense_2/bias
Ś
vc/dense_2/bias/Adam_1
VariableV2*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
shape:*
dtype0*
	container *
shared_name 
Ű
vc/dense_2/bias/Adam_1/AssignAssignvc/dense_2/bias/Adam_1(vc/dense_2/bias/Adam_1/Initializer/zeros*
T0*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(

vc/dense_2/bias/Adam_1/readIdentityvc/dense_2/bias/Adam_1*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *o:*
dtype0
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
valueB
 *wž?*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
Đ
%Adam/update_vf/dense/kernel/ApplyAdam	ApplyAdamvf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_54*
_output_shapes
:	<*
use_locking( *
use_nesterov( *
T0*"
_class
loc:@vf/dense/kernel
Â
#Adam/update_vf/dense/bias/ApplyAdam	ApplyAdamvf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_55*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
use_locking( *
T0*
use_nesterov( 
Ű
'Adam/update_vf/dense_1/kernel/ApplyAdam	ApplyAdamvf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_56*
T0*
use_locking( * 
_output_shapes
:
*
use_nesterov( *$
_class
loc:@vf/dense_1/kernel
Ě
%Adam/update_vf/dense_1/bias/ApplyAdam	ApplyAdamvf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_57*
T0*
use_locking( *"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_nesterov( 
Ú
'Adam/update_vf/dense_2/kernel/ApplyAdam	ApplyAdamvf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_58*$
_class
loc:@vf/dense_2/kernel*
use_nesterov( *
T0*
_output_shapes
:	*
use_locking( 
Ë
%Adam/update_vf/dense_2/bias/ApplyAdam	ApplyAdamvf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_59*"
_class
loc:@vf/dense_2/bias*
T0*
use_nesterov( *
_output_shapes
:*
use_locking( 
Đ
%Adam/update_vc/dense/kernel/ApplyAdam	ApplyAdamvc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_60*"
_class
loc:@vc/dense/kernel*
use_locking( *
use_nesterov( *
_output_shapes
:	<*
T0
Â
#Adam/update_vc/dense/bias/ApplyAdam	ApplyAdamvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_61*
use_nesterov( *
T0*
use_locking( *
_output_shapes	
:* 
_class
loc:@vc/dense/bias
Ű
'Adam/update_vc/dense_1/kernel/ApplyAdam	ApplyAdamvc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_62*
use_nesterov( *$
_class
loc:@vc/dense_1/kernel*
use_locking( *
T0* 
_output_shapes
:

Ě
%Adam/update_vc/dense_1/bias/ApplyAdam	ApplyAdamvc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_63*
_output_shapes	
:*
use_nesterov( *
T0*
use_locking( *"
_class
loc:@vc/dense_1/bias
Ú
'Adam/update_vc/dense_2/kernel/ApplyAdam	ApplyAdamvc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_64*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking( *
use_nesterov( 
Ë
%Adam/update_vc/dense_2/bias/ApplyAdam	ApplyAdamvc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_65*"
_class
loc:@vc/dense_2/bias*
use_locking( *
_output_shapes
:*
T0*
use_nesterov( 
Ô
Adam/mulMulbeta1_power/read
Adam/beta1$^Adam/update_vc/dense/bias/ApplyAdam&^Adam/update_vc/dense/kernel/ApplyAdam&^Adam/update_vc/dense_1/bias/ApplyAdam(^Adam/update_vc/dense_1/kernel/ApplyAdam&^Adam/update_vc/dense_2/bias/ApplyAdam(^Adam/update_vc/dense_2/kernel/ApplyAdam$^Adam/update_vf/dense/bias/ApplyAdam&^Adam/update_vf/dense/kernel/ApplyAdam&^Adam/update_vf/dense_1/bias/ApplyAdam(^Adam/update_vf/dense_1/kernel/ApplyAdam&^Adam/update_vf/dense_2/bias/ApplyAdam(^Adam/update_vf/dense_2/kernel/ApplyAdam* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: 
Ö

Adam/mul_1Mulbeta2_power/read
Adam/beta2$^Adam/update_vc/dense/bias/ApplyAdam&^Adam/update_vc/dense/kernel/ApplyAdam&^Adam/update_vc/dense_1/bias/ApplyAdam(^Adam/update_vc/dense_1/kernel/ApplyAdam&^Adam/update_vc/dense_2/bias/ApplyAdam(^Adam/update_vc/dense_2/kernel/ApplyAdam$^Adam/update_vf/dense/bias/ApplyAdam&^Adam/update_vf/dense/kernel/ApplyAdam&^Adam/update_vf/dense_1/bias/ApplyAdam(^Adam/update_vf/dense_1/kernel/ApplyAdam&^Adam/update_vf/dense_2/bias/ApplyAdam(^Adam/update_vf/dense_2/kernel/ApplyAdam* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0

AdamNoOp^Adam/Assign^Adam/Assign_1$^Adam/update_vc/dense/bias/ApplyAdam&^Adam/update_vc/dense/kernel/ApplyAdam&^Adam/update_vc/dense_1/bias/ApplyAdam(^Adam/update_vc/dense_1/kernel/ApplyAdam&^Adam/update_vc/dense_2/bias/ApplyAdam(^Adam/update_vc/dense_2/kernel/ApplyAdam$^Adam/update_vf/dense/bias/ApplyAdam&^Adam/update_vf/dense/kernel/ApplyAdam&^Adam/update_vf/dense_1/bias/ApplyAdam(^Adam/update_vf/dense_1/kernel/ApplyAdam&^Adam/update_vf/dense_2/bias/ApplyAdam(^Adam/update_vf/dense_2/kernel/ApplyAdam
j
Reshape_66/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
q

Reshape_66Reshapevf/dense/kernel/readReshape_66/shape*
Tshape0*
T0*
_output_shapes	
:x
j
Reshape_67/shapeConst^Adam*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
o

Reshape_67Reshapevf/dense/bias/readReshape_67/shape*
_output_shapes	
:*
Tshape0*
T0
j
Reshape_68/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
t

Reshape_68Reshapevf/dense_1/kernel/readReshape_68/shape*
Tshape0*
_output_shapes

:*
T0
j
Reshape_69/shapeConst^Adam*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
q

Reshape_69Reshapevf/dense_1/bias/readReshape_69/shape*
_output_shapes	
:*
Tshape0*
T0
j
Reshape_70/shapeConst^Adam*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
s

Reshape_70Reshapevf/dense_2/kernel/readReshape_70/shape*
_output_shapes	
:*
Tshape0*
T0
j
Reshape_71/shapeConst^Adam*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
p

Reshape_71Reshapevf/dense_2/bias/readReshape_71/shape*
_output_shapes
:*
Tshape0*
T0
j
Reshape_72/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
q

Reshape_72Reshapevc/dense/kernel/readReshape_72/shape*
_output_shapes	
:x*
T0*
Tshape0
j
Reshape_73/shapeConst^Adam*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
o

Reshape_73Reshapevc/dense/bias/readReshape_73/shape*
T0*
Tshape0*
_output_shapes	
:
j
Reshape_74/shapeConst^Adam*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
t

Reshape_74Reshapevc/dense_1/kernel/readReshape_74/shape*
_output_shapes

:*
Tshape0*
T0
j
Reshape_75/shapeConst^Adam*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
q

Reshape_75Reshapevc/dense_1/bias/readReshape_75/shape*
T0*
_output_shapes	
:*
Tshape0
j
Reshape_76/shapeConst^Adam*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
s

Reshape_76Reshapevc/dense_2/kernel/readReshape_76/shape*
Tshape0*
_output_shapes	
:*
T0
j
Reshape_77/shapeConst^Adam*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
p

Reshape_77Reshapevc/dense_2/bias/readReshape_77/shape*
Tshape0*
T0*
_output_shapes
:
V
concat_6/axisConst^Adam*
value	B : *
dtype0*
_output_shapes
: 
ď
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
Reshape_77concat_6/axis*

Tidx0*
N*
_output_shapes

:ü	*
T0
h
PyFunc_1PyFuncconcat_6*
_output_shapes
:*
Tout
2*
token
pyfunc_1*
Tin
2

Const_8Const^Adam*
_output_shapes
:*E
value<B:"0 <                  <                 *
dtype0
Z
split_2/split_dimConst^Adam*
value	B : *
_output_shapes
: *
dtype0
Ł
split_2SplitVPyFunc_1Const_8split_2/split_dim*D
_output_shapes2
0::::::::::::*
	num_split*

Tlen0*
T0
h
Reshape_78/shapeConst^Adam*
valueB"<      *
_output_shapes
:*
dtype0
h

Reshape_78Reshapesplit_2Reshape_78/shape*
T0*
_output_shapes
:	<*
Tshape0
b
Reshape_79/shapeConst^Adam*
dtype0*
valueB:*
_output_shapes
:
f

Reshape_79Reshape	split_2:1Reshape_79/shape*
T0*
_output_shapes	
:*
Tshape0
h
Reshape_80/shapeConst^Adam*
valueB"      *
_output_shapes
:*
dtype0
k

Reshape_80Reshape	split_2:2Reshape_80/shape*
T0* 
_output_shapes
:
*
Tshape0
b
Reshape_81/shapeConst^Adam*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_81Reshape	split_2:3Reshape_81/shape*
_output_shapes	
:*
Tshape0*
T0
h
Reshape_82/shapeConst^Adam*
dtype0*
valueB"      *
_output_shapes
:
j

Reshape_82Reshape	split_2:4Reshape_82/shape*
Tshape0*
_output_shapes
:	*
T0
a
Reshape_83/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:
e

Reshape_83Reshape	split_2:5Reshape_83/shape*
_output_shapes
:*
Tshape0*
T0
h
Reshape_84/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB"<      
j

Reshape_84Reshape	split_2:6Reshape_84/shape*
_output_shapes
:	<*
T0*
Tshape0
b
Reshape_85/shapeConst^Adam*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_85Reshape	split_2:7Reshape_85/shape*
_output_shapes	
:*
T0*
Tshape0
h
Reshape_86/shapeConst^Adam*
_output_shapes
:*
valueB"      *
dtype0
k

Reshape_86Reshape	split_2:8Reshape_86/shape*
Tshape0* 
_output_shapes
:
*
T0
b
Reshape_87/shapeConst^Adam*
valueB:*
_output_shapes
:*
dtype0
f

Reshape_87Reshape	split_2:9Reshape_87/shape*
_output_shapes	
:*
T0*
Tshape0
h
Reshape_88/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB"      
k

Reshape_88Reshape
split_2:10Reshape_88/shape*
T0*
_output_shapes
:	*
Tshape0
a
Reshape_89/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:
f

Reshape_89Reshape
split_2:11Reshape_89/shape*
Tshape0*
_output_shapes
:*
T0
Ś
Assign_7Assignvf/dense/kernel
Reshape_78*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
T0*
use_locking(*
validate_shape(

Assign_8Assignvf/dense/bias
Reshape_79*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0*
use_locking(*
validate_shape(
Ť
Assign_9Assignvf/dense_1/kernel
Reshape_80* 
_output_shapes
:
*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel
Ł
	Assign_10Assignvf/dense_1/bias
Reshape_81*
T0*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:
Ť
	Assign_11Assignvf/dense_2/kernel
Reshape_82*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
˘
	Assign_12Assignvf/dense_2/bias
Reshape_83*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
§
	Assign_13Assignvc/dense/kernel
Reshape_84*
validate_shape(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<*
use_locking(

	Assign_14Assignvc/dense/bias
Reshape_85*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:
Ź
	Assign_15Assignvc/dense_1/kernel
Reshape_86*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:

Ł
	Assign_16Assignvc/dense_1/bias
Reshape_87*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(*
validate_shape(
Ť
	Assign_17Assignvc/dense_2/kernel
Reshape_88*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(
˘
	Assign_18Assignvc/dense_2/bias
Reshape_89*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
¨
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


initNoOp^beta1_power/Assign^beta2_power/Assign^pi/dense/bias/Assign^pi/dense/kernel/Assign^pi/dense_1/bias/Assign^pi/dense_1/kernel/Assign^pi/dense_2/bias/Assign^pi/dense_2/kernel/Assign^pi/log_std/Assign^vc/dense/bias/Adam/Assign^vc/dense/bias/Adam_1/Assign^vc/dense/bias/Assign^vc/dense/kernel/Adam/Assign^vc/dense/kernel/Adam_1/Assign^vc/dense/kernel/Assign^vc/dense_1/bias/Adam/Assign^vc/dense_1/bias/Adam_1/Assign^vc/dense_1/bias/Assign^vc/dense_1/kernel/Adam/Assign ^vc/dense_1/kernel/Adam_1/Assign^vc/dense_1/kernel/Assign^vc/dense_2/bias/Adam/Assign^vc/dense_2/bias/Adam_1/Assign^vc/dense_2/bias/Assign^vc/dense_2/kernel/Adam/Assign ^vc/dense_2/kernel/Adam_1/Assign^vc/dense_2/kernel/Assign^vf/dense/bias/Adam/Assign^vf/dense/bias/Adam_1/Assign^vf/dense/bias/Assign^vf/dense/kernel/Adam/Assign^vf/dense/kernel/Adam_1/Assign^vf/dense/kernel/Assign^vf/dense_1/bias/Adam/Assign^vf/dense_1/bias/Adam_1/Assign^vf/dense_1/bias/Assign^vf/dense_1/kernel/Adam/Assign ^vf/dense_1/kernel/Adam_1/Assign^vf/dense_1/kernel/Assign^vf/dense_2/bias/Adam/Assign^vf/dense_2/bias/Adam_1/Assign^vf/dense_2/bias/Assign^vf/dense_2/kernel/Adam/Assign ^vf/dense_2/kernel/Adam_1/Assign^vf/dense_2/kernel/Assign
c
Reshape_90/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
q

Reshape_90Reshapepi/dense/kernel/readReshape_90/shape*
Tshape0*
T0*
_output_shapes	
:x
c
Reshape_91/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
o

Reshape_91Reshapepi/dense/bias/readReshape_91/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_92/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
t

Reshape_92Reshapepi/dense_1/kernel/readReshape_92/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_93/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
q

Reshape_93Reshapepi/dense_1/bias/readReshape_93/shape*
Tshape0*
_output_shapes	
:*
T0
c
Reshape_94/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
s

Reshape_94Reshapepi/dense_2/kernel/readReshape_94/shape*
T0*
_output_shapes	
:*
Tshape0
c
Reshape_95/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
p

Reshape_95Reshapepi/dense_2/bias/readReshape_95/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_96/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
k

Reshape_96Reshapepi/log_std/readReshape_96/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_97/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
q

Reshape_97Reshapevf/dense/kernel/readReshape_97/shape*
T0*
Tshape0*
_output_shapes	
:x
c
Reshape_98/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
o

Reshape_98Reshapevf/dense/bias/readReshape_98/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_99/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
t

Reshape_99Reshapevf/dense_1/kernel/readReshape_99/shape*
_output_shapes

:*
Tshape0*
T0
d
Reshape_100/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s
Reshape_100Reshapevf/dense_1/bias/readReshape_100/shape*
_output_shapes	
:*
Tshape0*
T0
d
Reshape_101/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
u
Reshape_101Reshapevf/dense_2/kernel/readReshape_101/shape*
_output_shapes	
:*
Tshape0*
T0
d
Reshape_102/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
r
Reshape_102Reshapevf/dense_2/bias/readReshape_102/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_103/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
s
Reshape_103Reshapevc/dense/kernel/readReshape_103/shape*
_output_shapes	
:x*
T0*
Tshape0
d
Reshape_104/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
q
Reshape_104Reshapevc/dense/bias/readReshape_104/shape*
T0*
_output_shapes	
:*
Tshape0
d
Reshape_105/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
v
Reshape_105Reshapevc/dense_1/kernel/readReshape_105/shape*
_output_shapes

:*
Tshape0*
T0
d
Reshape_106/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
s
Reshape_106Reshapevc/dense_1/bias/readReshape_106/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_107/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
u
Reshape_107Reshapevc/dense_2/kernel/readReshape_107/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_108/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
r
Reshape_108Reshapevc/dense_2/bias/readReshape_108/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_109/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
n
Reshape_109Reshapebeta1_power/readReshape_109/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_110/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
n
Reshape_110Reshapebeta2_power/readReshape_110/shape*
Tshape0*
_output_shapes
:*
T0
d
Reshape_111/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
x
Reshape_111Reshapevf/dense/kernel/Adam/readReshape_111/shape*
Tshape0*
T0*
_output_shapes	
:x
d
Reshape_112/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
z
Reshape_112Reshapevf/dense/kernel/Adam_1/readReshape_112/shape*
Tshape0*
T0*
_output_shapes	
:x
d
Reshape_113/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
v
Reshape_113Reshapevf/dense/bias/Adam/readReshape_113/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_114/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
x
Reshape_114Reshapevf/dense/bias/Adam_1/readReshape_114/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_115/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
{
Reshape_115Reshapevf/dense_1/kernel/Adam/readReshape_115/shape*
T0*
_output_shapes

:*
Tshape0
d
Reshape_116/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
}
Reshape_116Reshapevf/dense_1/kernel/Adam_1/readReshape_116/shape*
T0*
_output_shapes

:*
Tshape0
d
Reshape_117/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
x
Reshape_117Reshapevf/dense_1/bias/Adam/readReshape_117/shape*
T0*
Tshape0*
_output_shapes	
:
d
Reshape_118/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
z
Reshape_118Reshapevf/dense_1/bias/Adam_1/readReshape_118/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_119/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
z
Reshape_119Reshapevf/dense_2/kernel/Adam/readReshape_119/shape*
Tshape0*
_output_shapes	
:*
T0
d
Reshape_120/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
|
Reshape_120Reshapevf/dense_2/kernel/Adam_1/readReshape_120/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_121/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
w
Reshape_121Reshapevf/dense_2/bias/Adam/readReshape_121/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_122/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
y
Reshape_122Reshapevf/dense_2/bias/Adam_1/readReshape_122/shape*
T0*
_output_shapes
:*
Tshape0
d
Reshape_123/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
x
Reshape_123Reshapevc/dense/kernel/Adam/readReshape_123/shape*
_output_shapes	
:x*
Tshape0*
T0
d
Reshape_124/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
z
Reshape_124Reshapevc/dense/kernel/Adam_1/readReshape_124/shape*
_output_shapes	
:x*
Tshape0*
T0
d
Reshape_125/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
v
Reshape_125Reshapevc/dense/bias/Adam/readReshape_125/shape*
_output_shapes	
:*
Tshape0*
T0
d
Reshape_126/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
x
Reshape_126Reshapevc/dense/bias/Adam_1/readReshape_126/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_127/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
{
Reshape_127Reshapevc/dense_1/kernel/Adam/readReshape_127/shape*
_output_shapes

:*
T0*
Tshape0
d
Reshape_128/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
}
Reshape_128Reshapevc/dense_1/kernel/Adam_1/readReshape_128/shape*
Tshape0*
_output_shapes

:*
T0
d
Reshape_129/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
x
Reshape_129Reshapevc/dense_1/bias/Adam/readReshape_129/shape*
T0*
_output_shapes	
:*
Tshape0
d
Reshape_130/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
z
Reshape_130Reshapevc/dense_1/bias/Adam_1/readReshape_130/shape*
T0*
_output_shapes	
:*
Tshape0
d
Reshape_131/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
z
Reshape_131Reshapevc/dense_2/kernel/Adam/readReshape_131/shape*
Tshape0*
T0*
_output_shapes	
:
d
Reshape_132/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
|
Reshape_132Reshapevc/dense_2/kernel/Adam_1/readReshape_132/shape*
_output_shapes	
:*
T0*
Tshape0
d
Reshape_133/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
w
Reshape_133Reshapevc/dense_2/bias/Adam/readReshape_133/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_134/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
y
Reshape_134Reshapevc/dense_2/bias/Adam_1/readReshape_134/shape*
T0*
Tshape0*
_output_shapes
:
O
concat_7/axisConst*
value	B : *
_output_shapes
: *
dtype0

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
N-*
_output_shapes

:ô"*

Tidx0*
T0
h
PyFunc_2PyFuncconcat_7*
Tin
2*
_output_shapes
:*
token
pyfunc_2*
Tout
2

Const_9Const*Ě
valueÂBż-"´ <                     <                  <                        <   <                                 <   <                                *
_output_shapes
:-*
dtype0
S
split_3/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
Ş
split_3SplitVPyFunc_2Const_9split_3/split_dim*

Tlen0*
T0*
	num_split-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
b
Reshape_135/shapeConst*
dtype0*
valueB"<      *
_output_shapes
:
j
Reshape_135Reshapesplit_3Reshape_135/shape*
_output_shapes
:	<*
Tshape0*
T0
\
Reshape_136/shapeConst*
dtype0*
_output_shapes
:*
valueB:
h
Reshape_136Reshape	split_3:1Reshape_136/shape*
_output_shapes	
:*
Tshape0*
T0
b
Reshape_137/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
m
Reshape_137Reshape	split_3:2Reshape_137/shape*
T0* 
_output_shapes
:
*
Tshape0
\
Reshape_138/shapeConst*
dtype0*
_output_shapes
:*
valueB:
h
Reshape_138Reshape	split_3:3Reshape_138/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_139/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
l
Reshape_139Reshape	split_3:4Reshape_139/shape*
T0*
Tshape0*
_output_shapes
:	
[
Reshape_140/shapeConst*
_output_shapes
:*
dtype0*
valueB:
g
Reshape_140Reshape	split_3:5Reshape_140/shape*
Tshape0*
T0*
_output_shapes
:
[
Reshape_141/shapeConst*
_output_shapes
:*
valueB:*
dtype0
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
Reshape_142Reshape	split_3:7Reshape_142/shape*
T0*
_output_shapes
:	<*
Tshape0
\
Reshape_143/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_143Reshape	split_3:8Reshape_143/shape*
T0*
_output_shapes	
:*
Tshape0
b
Reshape_144/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_144Reshape	split_3:9Reshape_144/shape*
T0* 
_output_shapes
:
*
Tshape0
\
Reshape_145/shapeConst*
dtype0*
valueB:*
_output_shapes
:
i
Reshape_145Reshape
split_3:10Reshape_145/shape*
_output_shapes	
:*
T0*
Tshape0
b
Reshape_146/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
m
Reshape_146Reshape
split_3:11Reshape_146/shape*
T0*
Tshape0*
_output_shapes
:	
[
Reshape_147/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_147Reshape
split_3:12Reshape_147/shape*
T0*
_output_shapes
:*
Tshape0
b
Reshape_148/shapeConst*
dtype0*
valueB"<      *
_output_shapes
:
m
Reshape_148Reshape
split_3:13Reshape_148/shape*
_output_shapes
:	<*
Tshape0*
T0
\
Reshape_149/shapeConst*
_output_shapes
:*
valueB:*
dtype0
i
Reshape_149Reshape
split_3:14Reshape_149/shape*
_output_shapes	
:*
Tshape0*
T0
b
Reshape_150/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
n
Reshape_150Reshape
split_3:15Reshape_150/shape*
Tshape0* 
_output_shapes
:
*
T0
\
Reshape_151/shapeConst*
dtype0*
valueB:*
_output_shapes
:
i
Reshape_151Reshape
split_3:16Reshape_151/shape*
_output_shapes	
:*
Tshape0*
T0
b
Reshape_152/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_152Reshape
split_3:17Reshape_152/shape*
Tshape0*
_output_shapes
:	*
T0
[
Reshape_153/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_153Reshape
split_3:18Reshape_153/shape*
_output_shapes
:*
T0*
Tshape0
T
Reshape_154/shapeConst*
dtype0*
_output_shapes
: *
valueB 
d
Reshape_154Reshape
split_3:19Reshape_154/shape*
_output_shapes
: *
Tshape0*
T0
T
Reshape_155/shapeConst*
dtype0*
_output_shapes
: *
valueB 
d
Reshape_155Reshape
split_3:20Reshape_155/shape*
Tshape0*
T0*
_output_shapes
: 
b
Reshape_156/shapeConst*
dtype0*
_output_shapes
:*
valueB"<      
m
Reshape_156Reshape
split_3:21Reshape_156/shape*
T0*
Tshape0*
_output_shapes
:	<
b
Reshape_157/shapeConst*
valueB"<      *
dtype0*
_output_shapes
:
m
Reshape_157Reshape
split_3:22Reshape_157/shape*
_output_shapes
:	<*
Tshape0*
T0
\
Reshape_158/shapeConst*
valueB:*
_output_shapes
:*
dtype0
i
Reshape_158Reshape
split_3:23Reshape_158/shape*
_output_shapes	
:*
Tshape0*
T0
\
Reshape_159/shapeConst*
dtype0*
valueB:*
_output_shapes
:
i
Reshape_159Reshape
split_3:24Reshape_159/shape*
Tshape0*
T0*
_output_shapes	
:
b
Reshape_160/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
n
Reshape_160Reshape
split_3:25Reshape_160/shape*
Tshape0* 
_output_shapes
:
*
T0
b
Reshape_161/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
n
Reshape_161Reshape
split_3:26Reshape_161/shape* 
_output_shapes
:
*
Tshape0*
T0
\
Reshape_162/shapeConst*
dtype0*
_output_shapes
:*
valueB:
i
Reshape_162Reshape
split_3:27Reshape_162/shape*
_output_shapes	
:*
T0*
Tshape0
\
Reshape_163/shapeConst*
valueB:*
_output_shapes
:*
dtype0
i
Reshape_163Reshape
split_3:28Reshape_163/shape*
_output_shapes	
:*
Tshape0*
T0
b
Reshape_164/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_164Reshape
split_3:29Reshape_164/shape*
_output_shapes
:	*
Tshape0*
T0
b
Reshape_165/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
m
Reshape_165Reshape
split_3:30Reshape_165/shape*
T0*
Tshape0*
_output_shapes
:	
[
Reshape_166/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_166Reshape
split_3:31Reshape_166/shape*
Tshape0*
T0*
_output_shapes
:
[
Reshape_167/shapeConst*
dtype0*
_output_shapes
:*
valueB:
h
Reshape_167Reshape
split_3:32Reshape_167/shape*
Tshape0*
T0*
_output_shapes
:
b
Reshape_168/shapeConst*
_output_shapes
:*
dtype0*
valueB"<      
m
Reshape_168Reshape
split_3:33Reshape_168/shape*
_output_shapes
:	<*
Tshape0*
T0
b
Reshape_169/shapeConst*
_output_shapes
:*
valueB"<      *
dtype0
m
Reshape_169Reshape
split_3:34Reshape_169/shape*
T0*
_output_shapes
:	<*
Tshape0
\
Reshape_170/shapeConst*
_output_shapes
:*
valueB:*
dtype0
i
Reshape_170Reshape
split_3:35Reshape_170/shape*
_output_shapes	
:*
T0*
Tshape0
\
Reshape_171/shapeConst*
dtype0*
_output_shapes
:*
valueB:
i
Reshape_171Reshape
split_3:36Reshape_171/shape*
T0*
_output_shapes	
:*
Tshape0
b
Reshape_172/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
n
Reshape_172Reshape
split_3:37Reshape_172/shape* 
_output_shapes
:
*
Tshape0*
T0
b
Reshape_173/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
n
Reshape_173Reshape
split_3:38Reshape_173/shape*
Tshape0* 
_output_shapes
:
*
T0
\
Reshape_174/shapeConst*
_output_shapes
:*
dtype0*
valueB:
i
Reshape_174Reshape
split_3:39Reshape_174/shape*
_output_shapes	
:*
Tshape0*
T0
\
Reshape_175/shapeConst*
valueB:*
_output_shapes
:*
dtype0
i
Reshape_175Reshape
split_3:40Reshape_175/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_176/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_176Reshape
split_3:41Reshape_176/shape*
_output_shapes
:	*
Tshape0*
T0
b
Reshape_177/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
m
Reshape_177Reshape
split_3:42Reshape_177/shape*
Tshape0*
_output_shapes
:	*
T0
[
Reshape_178/shapeConst*
dtype0*
_output_shapes
:*
valueB:
h
Reshape_178Reshape
split_3:43Reshape_178/shape*
_output_shapes
:*
T0*
Tshape0
[
Reshape_179/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_179Reshape
split_3:44Reshape_179/shape*
Tshape0*
_output_shapes
:*
T0
¨
	Assign_19Assignpi/dense/kernelReshape_135*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
use_locking(
 
	Assign_20Assignpi/dense/biasReshape_136*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes	
:*
validate_shape(
­
	Assign_21Assignpi/dense_1/kernelReshape_137*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:

¤
	Assign_22Assignpi/dense_1/biasReshape_138*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:
Ź
	Assign_23Assignpi/dense_2/kernelReshape_139*
T0*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel
Ł
	Assign_24Assignpi/dense_2/biasReshape_140*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:

	Assign_25Assign
pi/log_stdReshape_141*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*
_class
loc:@pi/log_std
¨
	Assign_26Assignvf/dense/kernelReshape_142*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
T0*
validate_shape(
 
	Assign_27Assignvf/dense/biasReshape_143*
T0*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
­
	Assign_28Assignvf/dense_1/kernelReshape_144*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:

¤
	Assign_29Assignvf/dense_1/biasReshape_145*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:
Ź
	Assign_30Assignvf/dense_2/kernelReshape_146*
_output_shapes
:	*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0
Ł
	Assign_31Assignvf/dense_2/biasReshape_147*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
¨
	Assign_32Assignvc/dense/kernelReshape_148*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<
 
	Assign_33Assignvc/dense/biasReshape_149* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
­
	Assign_34Assignvc/dense_1/kernelReshape_150* 
_output_shapes
:
*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel
¤
	Assign_35Assignvc/dense_1/biasReshape_151*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(
Ź
	Assign_36Assignvc/dense_2/kernelReshape_152*
validate_shape(*
use_locking(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
T0
Ł
	Assign_37Assignvc/dense_2/biasReshape_153*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(*
_output_shapes
:

	Assign_38Assignbeta1_powerReshape_154* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
: 

	Assign_39Assignbeta2_powerReshape_155*
validate_shape(*
use_locking(*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias
­
	Assign_40Assignvf/dense/kernel/AdamReshape_156*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel
Ż
	Assign_41Assignvf/dense/kernel/Adam_1Reshape_157*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
T0*
validate_shape(
Ľ
	Assign_42Assignvf/dense/bias/AdamReshape_158*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
use_locking(*
T0*
validate_shape(
§
	Assign_43Assignvf/dense/bias/Adam_1Reshape_159*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
T0
˛
	Assign_44Assignvf/dense_1/kernel/AdamReshape_160*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
T0*
use_locking(
´
	Assign_45Assignvf/dense_1/kernel/Adam_1Reshape_161* 
_output_shapes
:
*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(
Š
	Assign_46Assignvf/dense_1/bias/AdamReshape_162*
validate_shape(*
use_locking(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0
Ť
	Assign_47Assignvf/dense_1/bias/Adam_1Reshape_163*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(
ą
	Assign_48Assignvf/dense_2/kernel/AdamReshape_164*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0
ł
	Assign_49Assignvf/dense_2/kernel/Adam_1Reshape_165*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
T0*
validate_shape(
¨
	Assign_50Assignvf/dense_2/bias/AdamReshape_166*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0*
use_locking(
Ş
	Assign_51Assignvf/dense_2/bias/Adam_1Reshape_167*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
­
	Assign_52Assignvc/dense/kernel/AdamReshape_168*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	<
Ż
	Assign_53Assignvc/dense/kernel/Adam_1Reshape_169*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
T0*
use_locking(*
validate_shape(
Ľ
	Assign_54Assignvc/dense/bias/AdamReshape_170*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(* 
_class
loc:@vc/dense/bias
§
	Assign_55Assignvc/dense/bias/Adam_1Reshape_171*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes	
:
˛
	Assign_56Assignvc/dense_1/kernel/AdamReshape_172*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:

´
	Assign_57Assignvc/dense_1/kernel/Adam_1Reshape_173*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

Š
	Assign_58Assignvc/dense_1/bias/AdamReshape_174*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias
Ť
	Assign_59Assignvc/dense_1/bias/Adam_1Reshape_175*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
ą
	Assign_60Assignvc/dense_2/kernel/AdamReshape_176*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(
ł
	Assign_61Assignvc/dense_2/kernel/Adam_1Reshape_177*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	
¨
	Assign_62Assignvc/dense_2/bias/AdamReshape_178*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0*
validate_shape(
Ş
	Assign_63Assignvc/dense_2/bias/Adam_1Reshape_179*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_2/bias
°
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
valueB Bmodel*
_output_shapes
: *
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
shape: *
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: *
_output_shapes
: 

save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_13f2d96d3bfc4606983d58994b68f8ff/part*
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
đ
save/SaveV2/tensor_namesConst*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-
˝
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ž
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
_output_shapes
: *
T0*'
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
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
ó
save/RestoreV2/tensor_namesConst*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
Ŕ
save/RestoreV2/shape_and_slicesConst*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-
ď
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::

save/AssignAssignbeta1_powersave/RestoreV2* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
: 
˘
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
validate_shape(*
T0*
_output_shapes
: *
use_locking(* 
_class
loc:@vc/dense/bias
Š
save/Assign_2Assignpi/dense/biassave/RestoreV2:2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
ą
save/Assign_3Assignpi/dense/kernelsave/RestoreV2:3*
use_locking(*
T0*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@pi/dense/kernel
­
save/Assign_4Assignpi/dense_1/biassave/RestoreV2:4*
T0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(
ś
save/Assign_5Assignpi/dense_1/kernelsave/RestoreV2:5*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ź
save/Assign_6Assignpi/dense_2/biassave/RestoreV2:6*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:
ľ
save/Assign_7Assignpi/dense_2/kernelsave/RestoreV2:7*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	
˘
save/Assign_8Assign
pi/log_stdsave/RestoreV2:8*
use_locking(*
T0*
_output_shapes
:*
_class
loc:@pi/log_std*
validate_shape(
Š
save/Assign_9Assignvc/dense/biassave/RestoreV2:9*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
°
save/Assign_10Assignvc/dense/bias/Adamsave/RestoreV2:10*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:
˛
save/Assign_11Assignvc/dense/bias/Adam_1save/RestoreV2:11*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
validate_shape(
ł
save/Assign_12Assignvc/dense/kernelsave/RestoreV2:12*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(*
_output_shapes
:	<*
validate_shape(
¸
save/Assign_13Assignvc/dense/kernel/Adamsave/RestoreV2:13*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0*
use_locking(
ş
save/Assign_14Assignvc/dense/kernel/Adam_1save/RestoreV2:14*
validate_shape(*
_output_shapes
:	<*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(
Ż
save/Assign_15Assignvc/dense_1/biassave/RestoreV2:15*
use_locking(*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(
´
save/Assign_16Assignvc/dense_1/bias/Adamsave/RestoreV2:16*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
ś
save/Assign_17Assignvc/dense_1/bias/Adam_1save/RestoreV2:17*
_output_shapes	
:*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
validate_shape(
¸
save/Assign_18Assignvc/dense_1/kernelsave/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:

˝
save/Assign_19Assignvc/dense_1/kernel/Adamsave/RestoreV2:19*
validate_shape(* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(
ż
save/Assign_20Assignvc/dense_1/kernel/Adam_1save/RestoreV2:20*
use_locking(* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0
Ž
save/Assign_21Assignvc/dense_2/biassave/RestoreV2:21*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0*
validate_shape(
ł
save/Assign_22Assignvc/dense_2/bias/Adamsave/RestoreV2:22*
_output_shapes
:*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
T0
ľ
save/Assign_23Assignvc/dense_2/bias/Adam_1save/RestoreV2:23*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias
ˇ
save/Assign_24Assignvc/dense_2/kernelsave/RestoreV2:24*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel
ź
save/Assign_25Assignvc/dense_2/kernel/Adamsave/RestoreV2:25*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	
ž
save/Assign_26Assignvc/dense_2/kernel/Adam_1save/RestoreV2:26*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(
Ť
save/Assign_27Assignvf/dense/biassave/RestoreV2:27* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:
°
save/Assign_28Assignvf/dense/bias/Adamsave/RestoreV2:28*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vf/dense/bias
˛
save/Assign_29Assignvf/dense/bias/Adam_1save/RestoreV2:29* 
_class
loc:@vf/dense/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
ł
save/Assign_30Assignvf/dense/kernelsave/RestoreV2:30*
_output_shapes
:	<*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel
¸
save/Assign_31Assignvf/dense/kernel/Adamsave/RestoreV2:31*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
validate_shape(
ş
save/Assign_32Assignvf/dense/kernel/Adam_1save/RestoreV2:32*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	<*
use_locking(
Ż
save/Assign_33Assignvf/dense_1/biassave/RestoreV2:33*
T0*
use_locking(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
validate_shape(
´
save/Assign_34Assignvf/dense_1/bias/Adamsave/RestoreV2:34*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:
ś
save/Assign_35Assignvf/dense_1/bias/Adam_1save/RestoreV2:35*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0
¸
save/Assign_36Assignvf/dense_1/kernelsave/RestoreV2:36*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:
*
validate_shape(
˝
save/Assign_37Assignvf/dense_1/kernel/Adamsave/RestoreV2:37*
use_locking(* 
_output_shapes
:
*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel
ż
save/Assign_38Assignvf/dense_1/kernel/Adam_1save/RestoreV2:38*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0*
validate_shape(
Ž
save/Assign_39Assignvf/dense_2/biassave/RestoreV2:39*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
ł
save/Assign_40Assignvf/dense_2/bias/Adamsave/RestoreV2:40*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
use_locking(
ľ
save/Assign_41Assignvf/dense_2/bias/Adam_1save/RestoreV2:41*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
T0
ˇ
save/Assign_42Assignvf/dense_2/kernelsave/RestoreV2:42*
validate_shape(*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@vf/dense_2/kernel
ź
save/Assign_43Assignvf/dense_2/kernel/Adamsave/RestoreV2:43*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0
ž
save/Assign_44Assignvf/dense_2/kernel/Adam_1save/RestoreV2:44*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
_output_shapes
: *
shape: *
dtype0

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_b727897d9c154001b77e3f4d1d391761/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_1/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
ň
save_1/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ż
save_1/SaveV2/shape_and_slicesConst*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-
ś
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: *
T0
Ł
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
N*
T0*
_output_shapes
:*

axis 

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
_output_shapes
: *
T0
ő
save_1/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
Â
!save_1/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
÷
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
˘
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
_output_shapes
: *
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
use_locking(
Ś
save_1/Assign_1Assignbeta2_powersave_1/RestoreV2:1*
use_locking(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0*
validate_shape(
­
save_1/Assign_2Assignpi/dense/biassave_1/RestoreV2:2*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0*
validate_shape(
ľ
save_1/Assign_3Assignpi/dense/kernelsave_1/RestoreV2:3*
_output_shapes
:	<*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(
ą
save_1/Assign_4Assignpi/dense_1/biassave_1/RestoreV2:4*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(
ş
save_1/Assign_5Assignpi/dense_1/kernelsave_1/RestoreV2:5*
T0* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(
°
save_1/Assign_6Assignpi/dense_2/biassave_1/RestoreV2:6*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(
š
save_1/Assign_7Assignpi/dense_2/kernelsave_1/RestoreV2:7*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes
:	*
T0*
validate_shape(
Ś
save_1/Assign_8Assign
pi/log_stdsave_1/RestoreV2:8*
T0*
use_locking(*
_class
loc:@pi/log_std*
_output_shapes
:*
validate_shape(
­
save_1/Assign_9Assignvc/dense/biassave_1/RestoreV2:9* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(
´
save_1/Assign_10Assignvc/dense/bias/Adamsave_1/RestoreV2:10* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(
ś
save_1/Assign_11Assignvc/dense/bias/Adam_1save_1/RestoreV2:11*
use_locking(*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:
ˇ
save_1/Assign_12Assignvc/dense/kernelsave_1/RestoreV2:12*
use_locking(*
_output_shapes
:	<*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel
ź
save_1/Assign_13Assignvc/dense/kernel/Adamsave_1/RestoreV2:13*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	<
ž
save_1/Assign_14Assignvc/dense/kernel/Adam_1save_1/RestoreV2:14*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	<
ł
save_1/Assign_15Assignvc/dense_1/biassave_1/RestoreV2:15*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(
¸
save_1/Assign_16Assignvc/dense_1/bias/Adamsave_1/RestoreV2:16*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(
ş
save_1/Assign_17Assignvc/dense_1/bias/Adam_1save_1/RestoreV2:17*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(
ź
save_1/Assign_18Assignvc/dense_1/kernelsave_1/RestoreV2:18* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
T0*
validate_shape(
Á
save_1/Assign_19Assignvc/dense_1/kernel/Adamsave_1/RestoreV2:19*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
Ă
save_1/Assign_20Assignvc/dense_1/kernel/Adam_1save_1/RestoreV2:20*
validate_shape(* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(
˛
save_1/Assign_21Assignvc/dense_2/biassave_1/RestoreV2:21*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias
ˇ
save_1/Assign_22Assignvc/dense_2/bias/Adamsave_1/RestoreV2:22*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense_2/bias
š
save_1/Assign_23Assignvc/dense_2/bias/Adam_1save_1/RestoreV2:23*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias
ť
save_1/Assign_24Assignvc/dense_2/kernelsave_1/RestoreV2:24*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	
Ŕ
save_1/Assign_25Assignvc/dense_2/kernel/Adamsave_1/RestoreV2:25*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
T0
Â
save_1/Assign_26Assignvc/dense_2/kernel/Adam_1save_1/RestoreV2:26*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
Ż
save_1/Assign_27Assignvf/dense/biassave_1/RestoreV2:27*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_1/Assign_28Assignvf/dense/bias/Adamsave_1/RestoreV2:28* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(
ś
save_1/Assign_29Assignvf/dense/bias/Adam_1save_1/RestoreV2:29*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias
ˇ
save_1/Assign_30Assignvf/dense/kernelsave_1/RestoreV2:30*
use_locking(*
T0*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@vf/dense/kernel
ź
save_1/Assign_31Assignvf/dense/kernel/Adamsave_1/RestoreV2:31*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
validate_shape(
ž
save_1/Assign_32Assignvf/dense/kernel/Adam_1save_1/RestoreV2:32*
use_locking(*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	<
ł
save_1/Assign_33Assignvf/dense_1/biassave_1/RestoreV2:33*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vf/dense_1/bias
¸
save_1/Assign_34Assignvf/dense_1/bias/Adamsave_1/RestoreV2:34*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias
ş
save_1/Assign_35Assignvf/dense_1/bias/Adam_1save_1/RestoreV2:35*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0
ź
save_1/Assign_36Assignvf/dense_1/kernelsave_1/RestoreV2:36* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(*
use_locking(
Á
save_1/Assign_37Assignvf/dense_1/kernel/Adamsave_1/RestoreV2:37*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:

Ă
save_1/Assign_38Assignvf/dense_1/kernel/Adam_1save_1/RestoreV2:38* 
_output_shapes
:
*
T0*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(
˛
save_1/Assign_39Assignvf/dense_2/biassave_1/RestoreV2:39*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
use_locking(
ˇ
save_1/Assign_40Assignvf/dense_2/bias/Adamsave_1/RestoreV2:40*
validate_shape(*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:
š
save_1/Assign_41Assignvf/dense_2/bias/Adam_1save_1/RestoreV2:41*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ť
save_1/Assign_42Assignvf/dense_2/kernelsave_1/RestoreV2:42*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	*
T0
Ŕ
save_1/Assign_43Assignvf/dense_2/kernel/Adamsave_1/RestoreV2:43*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
Â
save_1/Assign_44Assignvf/dense_2/kernel/Adam_1save_1/RestoreV2:44*
T0*
use_locking(*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
ç
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
shape: *
dtype0*
_output_shapes
: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
_output_shapes
: *
shape: *
dtype0

save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_dd110bac94cc4df299cde4eb914e0485/part*
dtype0*
_output_shapes
: 
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
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
ň
save_2/SaveV2/tensor_namesConst*
_output_shapes
:-*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ż
save_2/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ś
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*)
_class
loc:@save_2/ShardedFilename*
_output_shapes
: 
Ł
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
_output_shapes
:*
N*
T0*

axis 

save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(

save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
T0*
_output_shapes
: 
ő
save_2/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Â
!save_2/RestoreV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-*
dtype0
÷
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
˘
save_2/AssignAssignbeta1_powersave_2/RestoreV2*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: 
Ś
save_2/Assign_1Assignbeta2_powersave_2/RestoreV2:1*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(
­
save_2/Assign_2Assignpi/dense/biassave_2/RestoreV2:2*
use_locking(*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:
ľ
save_2/Assign_3Assignpi/dense/kernelsave_2/RestoreV2:3*
T0*
validate_shape(*
_output_shapes
:	<*
use_locking(*"
_class
loc:@pi/dense/kernel
ą
save_2/Assign_4Assignpi/dense_1/biassave_2/RestoreV2:4*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
validate_shape(*
use_locking(
ş
save_2/Assign_5Assignpi/dense_1/kernelsave_2/RestoreV2:5* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
°
save_2/Assign_6Assignpi/dense_2/biassave_2/RestoreV2:6*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(
š
save_2/Assign_7Assignpi/dense_2/kernelsave_2/RestoreV2:7*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes
:	*
T0
Ś
save_2/Assign_8Assign
pi/log_stdsave_2/RestoreV2:8*
validate_shape(*
_class
loc:@pi/log_std*
T0*
use_locking(*
_output_shapes
:
­
save_2/Assign_9Assignvc/dense/biassave_2/RestoreV2:9*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(* 
_class
loc:@vc/dense/bias
´
save_2/Assign_10Assignvc/dense/bias/Adamsave_2/RestoreV2:10*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(* 
_class
loc:@vc/dense/bias
ś
save_2/Assign_11Assignvc/dense/bias/Adam_1save_2/RestoreV2:11*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
use_locking(*
T0*
validate_shape(
ˇ
save_2/Assign_12Assignvc/dense/kernelsave_2/RestoreV2:12*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	<*
T0*
use_locking(
ź
save_2/Assign_13Assignvc/dense/kernel/Adamsave_2/RestoreV2:13*
_output_shapes
:	<*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel
ž
save_2/Assign_14Assignvc/dense/kernel/Adam_1save_2/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
validate_shape(*
use_locking(*
T0
ł
save_2/Assign_15Assignvc/dense_1/biassave_2/RestoreV2:15*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
¸
save_2/Assign_16Assignvc/dense_1/bias/Adamsave_2/RestoreV2:16*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
ş
save_2/Assign_17Assignvc/dense_1/bias/Adam_1save_2/RestoreV2:17*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(*"
_class
loc:@vc/dense_1/bias
ź
save_2/Assign_18Assignvc/dense_1/kernelsave_2/RestoreV2:18*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:

Á
save_2/Assign_19Assignvc/dense_1/kernel/Adamsave_2/RestoreV2:19* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(
Ă
save_2/Assign_20Assignvc/dense_1/kernel/Adam_1save_2/RestoreV2:20*
validate_shape(*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0
˛
save_2/Assign_21Assignvc/dense_2/biassave_2/RestoreV2:21*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:
ˇ
save_2/Assign_22Assignvc/dense_2/bias/Adamsave_2/RestoreV2:22*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0
š
save_2/Assign_23Assignvc/dense_2/bias/Adam_1save_2/RestoreV2:23*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0
ť
save_2/Assign_24Assignvc/dense_2/kernelsave_2/RestoreV2:24*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	
Ŕ
save_2/Assign_25Assignvc/dense_2/kernel/Adamsave_2/RestoreV2:25*
T0*
use_locking(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
Â
save_2/Assign_26Assignvc/dense_2/kernel/Adam_1save_2/RestoreV2:26*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0
Ż
save_2/Assign_27Assignvf/dense/biassave_2/RestoreV2:27*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
´
save_2/Assign_28Assignvf/dense/bias/Adamsave_2/RestoreV2:28*
use_locking(*
_output_shapes	
:*
validate_shape(* 
_class
loc:@vf/dense/bias*
T0
ś
save_2/Assign_29Assignvf/dense/bias/Adam_1save_2/RestoreV2:29*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
ˇ
save_2/Assign_30Assignvf/dense/kernelsave_2/RestoreV2:30*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@vf/dense/kernel*
T0*
use_locking(
ź
save_2/Assign_31Assignvf/dense/kernel/Adamsave_2/RestoreV2:31*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
T0
ž
save_2/Assign_32Assignvf/dense/kernel/Adam_1save_2/RestoreV2:32*
_output_shapes
:	<*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel
ł
save_2/Assign_33Assignvf/dense_1/biassave_2/RestoreV2:33*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0
¸
save_2/Assign_34Assignvf/dense_1/bias/Adamsave_2/RestoreV2:34*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
ş
save_2/Assign_35Assignvf/dense_1/bias/Adam_1save_2/RestoreV2:35*
T0*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(
ź
save_2/Assign_36Assignvf/dense_1/kernelsave_2/RestoreV2:36*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Á
save_2/Assign_37Assignvf/dense_1/kernel/Adamsave_2/RestoreV2:37* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(*
use_locking(
Ă
save_2/Assign_38Assignvf/dense_1/kernel/Adam_1save_2/RestoreV2:38*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
˛
save_2/Assign_39Assignvf/dense_2/biassave_2/RestoreV2:39*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:
ˇ
save_2/Assign_40Assignvf/dense_2/bias/Adamsave_2/RestoreV2:40*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(
š
save_2/Assign_41Assignvf/dense_2/bias/Adam_1save_2/RestoreV2:41*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(
ť
save_2/Assign_42Assignvf/dense_2/kernelsave_2/RestoreV2:42*$
_class
loc:@vf/dense_2/kernel*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(
Ŕ
save_2/Assign_43Assignvf/dense_2/kernel/Adamsave_2/RestoreV2:43*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
Â
save_2/Assign_44Assignvf/dense_2/kernel/Adam_1save_2/RestoreV2:44*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
ç
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_40^save_2/Assign_41^save_2/Assign_42^save_2/Assign_43^save_2/Assign_44^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
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

save_3/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_9ad24d558fab466c82b2e44003f40da4/part*
_output_shapes
: 
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
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

save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
ň
save_3/SaveV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
ż
save_3/SaveV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
ś
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
_output_shapes
: *)
_class
loc:@save_3/ShardedFilename*
T0
Ł
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
N*
T0*

axis *
_output_shapes
:

save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(

save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
T0*
_output_shapes
: 
ő
save_3/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
Â
!save_3/RestoreV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
÷
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
˘
save_3/AssignAssignbeta1_powersave_3/RestoreV2*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(
Ś
save_3/Assign_1Assignbeta2_powersave_3/RestoreV2:1*
T0*
use_locking(*
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias
­
save_3/Assign_2Assignpi/dense/biassave_3/RestoreV2:2*
T0*
use_locking(*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
validate_shape(
ľ
save_3/Assign_3Assignpi/dense/kernelsave_3/RestoreV2:3*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(*
T0
ą
save_3/Assign_4Assignpi/dense_1/biassave_3/RestoreV2:4*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:
ş
save_3/Assign_5Assignpi/dense_1/kernelsave_3/RestoreV2:5*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:

°
save_3/Assign_6Assignpi/dense_2/biassave_3/RestoreV2:6*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias
š
save_3/Assign_7Assignpi/dense_2/kernelsave_3/RestoreV2:7*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ś
save_3/Assign_8Assign
pi/log_stdsave_3/RestoreV2:8*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
­
save_3/Assign_9Assignvc/dense/biassave_3/RestoreV2:9*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:*
validate_shape(
´
save_3/Assign_10Assignvc/dense/bias/Adamsave_3/RestoreV2:10*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vc/dense/bias
ś
save_3/Assign_11Assignvc/dense/bias/Adam_1save_3/RestoreV2:11*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(
ˇ
save_3/Assign_12Assignvc/dense/kernelsave_3/RestoreV2:12*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel
ź
save_3/Assign_13Assignvc/dense/kernel/Adamsave_3/RestoreV2:13*
_output_shapes
:	<*
T0*
validate_shape(*"
_class
loc:@vc/dense/kernel*
use_locking(
ž
save_3/Assign_14Assignvc/dense/kernel/Adam_1save_3/RestoreV2:14*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<*
validate_shape(
ł
save_3/Assign_15Assignvc/dense_1/biassave_3/RestoreV2:15*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
¸
save_3/Assign_16Assignvc/dense_1/bias/Adamsave_3/RestoreV2:16*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(
ş
save_3/Assign_17Assignvc/dense_1/bias/Adam_1save_3/RestoreV2:17*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(*
T0
ź
save_3/Assign_18Assignvc/dense_1/kernelsave_3/RestoreV2:18* 
_output_shapes
:
*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vc/dense_1/kernel
Á
save_3/Assign_19Assignvc/dense_1/kernel/Adamsave_3/RestoreV2:19* 
_output_shapes
:
*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
use_locking(
Ă
save_3/Assign_20Assignvc/dense_1/kernel/Adam_1save_3/RestoreV2:20*
T0*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
˛
save_3/Assign_21Assignvc/dense_2/biassave_3/RestoreV2:21*
T0*"
_class
loc:@vc/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
ˇ
save_3/Assign_22Assignvc/dense_2/bias/Adamsave_3/RestoreV2:22*"
_class
loc:@vc/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
š
save_3/Assign_23Assignvc/dense_2/bias/Adam_1save_3/RestoreV2:23*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(
ť
save_3/Assign_24Assignvc/dense_2/kernelsave_3/RestoreV2:24*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
T0*
use_locking(*
validate_shape(
Ŕ
save_3/Assign_25Assignvc/dense_2/kernel/Adamsave_3/RestoreV2:25*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
Â
save_3/Assign_26Assignvc/dense_2/kernel/Adam_1save_3/RestoreV2:26*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
T0*
use_locking(
Ż
save_3/Assign_27Assignvf/dense/biassave_3/RestoreV2:27*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
use_locking(*
T0
´
save_3/Assign_28Assignvf/dense/bias/Adamsave_3/RestoreV2:28*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
use_locking(*
T0*
validate_shape(
ś
save_3/Assign_29Assignvf/dense/bias/Adam_1save_3/RestoreV2:29*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
ˇ
save_3/Assign_30Assignvf/dense/kernelsave_3/RestoreV2:30*
use_locking(*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	<
ź
save_3/Assign_31Assignvf/dense/kernel/Adamsave_3/RestoreV2:31*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	<*
T0
ž
save_3/Assign_32Assignvf/dense/kernel/Adam_1save_3/RestoreV2:32*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	<*
T0
ł
save_3/Assign_33Assignvf/dense_1/biassave_3/RestoreV2:33*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(*
T0
¸
save_3/Assign_34Assignvf/dense_1/bias/Adamsave_3/RestoreV2:34*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
ş
save_3/Assign_35Assignvf/dense_1/bias/Adam_1save_3/RestoreV2:35*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:
ź
save_3/Assign_36Assignvf/dense_1/kernelsave_3/RestoreV2:36* 
_output_shapes
:
*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel
Á
save_3/Assign_37Assignvf/dense_1/kernel/Adamsave_3/RestoreV2:37*
T0* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(
Ă
save_3/Assign_38Assignvf/dense_1/kernel/Adam_1save_3/RestoreV2:38*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0
˛
save_3/Assign_39Assignvf/dense_2/biassave_3/RestoreV2:39*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias
ˇ
save_3/Assign_40Assignvf/dense_2/bias/Adamsave_3/RestoreV2:40*
_output_shapes
:*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(*
T0
š
save_3/Assign_41Assignvf/dense_2/bias/Adam_1save_3/RestoreV2:41*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias
ť
save_3/Assign_42Assignvf/dense_2/kernelsave_3/RestoreV2:42*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
T0
Ŕ
save_3/Assign_43Assignvf/dense_2/kernel/Adamsave_3/RestoreV2:43*
T0*
validate_shape(*
_output_shapes
:	*
use_locking(*$
_class
loc:@vf/dense_2/kernel
Â
save_3/Assign_44Assignvf/dense_2/kernel/Adam_1save_3/RestoreV2:44*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	
ç
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_5^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9
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
shape: *
_output_shapes
: *
dtype0
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
shape: *
_output_shapes
: *
dtype0

save_4/StringJoin/inputs_1Const*<
value3B1 B+_temp_5d20d95a2f364cbcb51aa03f314670f5/part*
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
save_4/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
ň
save_4/SaveV2/tensor_namesConst*
_output_shapes
:-*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
ż
save_4/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ś
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*)
_class
loc:@save_4/ShardedFilename*
_output_shapes
: 
Ł
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*
N*

axis *
T0*
_output_shapes
:

save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(

save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
_output_shapes
: *
T0
ő
save_4/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Â
!save_4/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
÷
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
˘
save_4/AssignAssignbeta1_powersave_4/RestoreV2*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(
Ś
save_4/Assign_1Assignbeta2_powersave_4/RestoreV2:1*
T0*
use_locking(*
_output_shapes
: *
validate_shape(* 
_class
loc:@vc/dense/bias
­
save_4/Assign_2Assignpi/dense/biassave_4/RestoreV2:2* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(
ľ
save_4/Assign_3Assignpi/dense/kernelsave_4/RestoreV2:3*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
validate_shape(
ą
save_4/Assign_4Assignpi/dense_1/biassave_4/RestoreV2:4*
_output_shapes	
:*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(
ş
save_4/Assign_5Assignpi/dense_1/kernelsave_4/RestoreV2:5*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(*
use_locking(* 
_output_shapes
:

°
save_4/Assign_6Assignpi/dense_2/biassave_4/RestoreV2:6*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
š
save_4/Assign_7Assignpi/dense_2/kernelsave_4/RestoreV2:7*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
Ś
save_4/Assign_8Assign
pi/log_stdsave_4/RestoreV2:8*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(
­
save_4/Assign_9Assignvc/dense/biassave_4/RestoreV2:9*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@vc/dense/bias*
T0
´
save_4/Assign_10Assignvc/dense/bias/Adamsave_4/RestoreV2:10*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
ś
save_4/Assign_11Assignvc/dense/bias/Adam_1save_4/RestoreV2:11* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
ˇ
save_4/Assign_12Assignvc/dense/kernelsave_4/RestoreV2:12*
validate_shape(*"
_class
loc:@vc/dense/kernel*
use_locking(*
_output_shapes
:	<*
T0
ź
save_4/Assign_13Assignvc/dense/kernel/Adamsave_4/RestoreV2:13*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(*
T0
ž
save_4/Assign_14Assignvc/dense/kernel/Adam_1save_4/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	<*
T0*
use_locking(
ł
save_4/Assign_15Assignvc/dense_1/biassave_4/RestoreV2:15*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:*
T0
¸
save_4/Assign_16Assignvc/dense_1/bias/Adamsave_4/RestoreV2:16*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
ş
save_4/Assign_17Assignvc/dense_1/bias/Adam_1save_4/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_4/Assign_18Assignvc/dense_1/kernelsave_4/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(
Á
save_4/Assign_19Assignvc/dense_1/kernel/Adamsave_4/RestoreV2:19*
T0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(
Ă
save_4/Assign_20Assignvc/dense_1/kernel/Adam_1save_4/RestoreV2:20*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:

˛
save_4/Assign_21Assignvc/dense_2/biassave_4/RestoreV2:21*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias
ˇ
save_4/Assign_22Assignvc/dense_2/bias/Adamsave_4/RestoreV2:22*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
use_locking(
š
save_4/Assign_23Assignvc/dense_2/bias/Adam_1save_4/RestoreV2:23*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_2/bias
ť
save_4/Assign_24Assignvc/dense_2/kernelsave_4/RestoreV2:24*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
Ŕ
save_4/Assign_25Assignvc/dense_2/kernel/Adamsave_4/RestoreV2:25*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	*
validate_shape(*
use_locking(
Â
save_4/Assign_26Assignvc/dense_2/kernel/Adam_1save_4/RestoreV2:26*
use_locking(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0
Ż
save_4/Assign_27Assignvf/dense/biassave_4/RestoreV2:27*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
´
save_4/Assign_28Assignvf/dense/bias/Adamsave_4/RestoreV2:28*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
ś
save_4/Assign_29Assignvf/dense/bias/Adam_1save_4/RestoreV2:29*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(* 
_class
loc:@vf/dense/bias
ˇ
save_4/Assign_30Assignvf/dense/kernelsave_4/RestoreV2:30*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	<
ź
save_4/Assign_31Assignvf/dense/kernel/Adamsave_4/RestoreV2:31*
T0*
validate_shape(*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	<
ž
save_4/Assign_32Assignvf/dense/kernel/Adam_1save_4/RestoreV2:32*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	<*
T0
ł
save_4/Assign_33Assignvf/dense_1/biassave_4/RestoreV2:33*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(*
T0
¸
save_4/Assign_34Assignvf/dense_1/bias/Adamsave_4/RestoreV2:34*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
use_locking(
ş
save_4/Assign_35Assignvf/dense_1/bias/Adam_1save_4/RestoreV2:35*
T0*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(
ź
save_4/Assign_36Assignvf/dense_1/kernelsave_4/RestoreV2:36*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:

Á
save_4/Assign_37Assignvf/dense_1/kernel/Adamsave_4/RestoreV2:37*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
validate_shape(*
T0
Ă
save_4/Assign_38Assignvf/dense_1/kernel/Adam_1save_4/RestoreV2:38*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:
*
validate_shape(
˛
save_4/Assign_39Assignvf/dense_2/biassave_4/RestoreV2:39*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
ˇ
save_4/Assign_40Assignvf/dense_2/bias/Adamsave_4/RestoreV2:40*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0
š
save_4/Assign_41Assignvf/dense_2/bias/Adam_1save_4/RestoreV2:41*
_output_shapes
:*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
T0*
use_locking(
ť
save_4/Assign_42Assignvf/dense_2/kernelsave_4/RestoreV2:42*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0
Ŕ
save_4/Assign_43Assignvf/dense_2/kernel/Adamsave_4/RestoreV2:43*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
Â
save_4/Assign_44Assignvf/dense_2/kernel/Adam_1save_4/RestoreV2:44*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
T0*
validate_shape(
ç
save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_2^save_4/Assign_20^save_4/Assign_21^save_4/Assign_22^save_4/Assign_23^save_4/Assign_24^save_4/Assign_25^save_4/Assign_26^save_4/Assign_27^save_4/Assign_28^save_4/Assign_29^save_4/Assign_3^save_4/Assign_30^save_4/Assign_31^save_4/Assign_32^save_4/Assign_33^save_4/Assign_34^save_4/Assign_35^save_4/Assign_36^save_4/Assign_37^save_4/Assign_38^save_4/Assign_39^save_4/Assign_4^save_4/Assign_40^save_4/Assign_41^save_4/Assign_42^save_4/Assign_43^save_4/Assign_44^save_4/Assign_5^save_4/Assign_6^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard
[
save_5/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
dtype0*
shape: *
_output_shapes
: 

save_5/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_42a5655c0a594a0b844e0230cfb30efa/part
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
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
ň
save_5/SaveV2/tensor_namesConst*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-
ż
save_5/SaveV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ś
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*
T0*)
_class
loc:@save_5/ShardedFilename*
_output_shapes
: 
Ł
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*

axis *
_output_shapes
:*
T0*
N

save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(

save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
_output_shapes
: *
T0
ő
save_5/RestoreV2/tensor_namesConst*
_output_shapes
:-*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Â
!save_5/RestoreV2/shape_and_slicesConst*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-
÷
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
˘
save_5/AssignAssignbeta1_powersave_5/RestoreV2*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(
Ś
save_5/Assign_1Assignbeta2_powersave_5/RestoreV2:1*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: 
­
save_5/Assign_2Assignpi/dense/biassave_5/RestoreV2:2*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0* 
_class
loc:@pi/dense/bias
ľ
save_5/Assign_3Assignpi/dense/kernelsave_5/RestoreV2:3*
_output_shapes
:	<*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
ą
save_5/Assign_4Assignpi/dense_1/biassave_5/RestoreV2:4*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0
ş
save_5/Assign_5Assignpi/dense_1/kernelsave_5/RestoreV2:5*
use_locking(* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0
°
save_5/Assign_6Assignpi/dense_2/biassave_5/RestoreV2:6*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias
š
save_5/Assign_7Assignpi/dense_2/kernelsave_5/RestoreV2:7*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
T0
Ś
save_5/Assign_8Assign
pi/log_stdsave_5/RestoreV2:8*
use_locking(*
_output_shapes
:*
_class
loc:@pi/log_std*
T0*
validate_shape(
­
save_5/Assign_9Assignvc/dense/biassave_5/RestoreV2:9*
use_locking(*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
validate_shape(
´
save_5/Assign_10Assignvc/dense/bias/Adamsave_5/RestoreV2:10*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
use_locking(*
validate_shape(
ś
save_5/Assign_11Assignvc/dense/bias/Adam_1save_5/RestoreV2:11*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0
ˇ
save_5/Assign_12Assignvc/dense/kernelsave_5/RestoreV2:12*
validate_shape(*
_output_shapes
:	<*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(
ź
save_5/Assign_13Assignvc/dense/kernel/Adamsave_5/RestoreV2:13*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
validate_shape(*
T0
ž
save_5/Assign_14Assignvc/dense/kernel/Adam_1save_5/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	<*
use_locking(*
T0
ł
save_5/Assign_15Assignvc/dense_1/biassave_5/RestoreV2:15*
validate_shape(*
use_locking(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0
¸
save_5/Assign_16Assignvc/dense_1/bias/Adamsave_5/RestoreV2:16*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(
ş
save_5/Assign_17Assignvc/dense_1/bias/Adam_1save_5/RestoreV2:17*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(
ź
save_5/Assign_18Assignvc/dense_1/kernelsave_5/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:

Á
save_5/Assign_19Assignvc/dense_1/kernel/Adamsave_5/RestoreV2:19*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(
Ă
save_5/Assign_20Assignvc/dense_1/kernel/Adam_1save_5/RestoreV2:20*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

˛
save_5/Assign_21Assignvc/dense_2/biassave_5/RestoreV2:21*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
ˇ
save_5/Assign_22Assignvc/dense_2/bias/Adamsave_5/RestoreV2:22*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias
š
save_5/Assign_23Assignvc/dense_2/bias/Adam_1save_5/RestoreV2:23*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ť
save_5/Assign_24Assignvc/dense_2/kernelsave_5/RestoreV2:24*
validate_shape(*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(
Ŕ
save_5/Assign_25Assignvc/dense_2/kernel/Adamsave_5/RestoreV2:25*
validate_shape(*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
use_locking(
Â
save_5/Assign_26Assignvc/dense_2/kernel/Adam_1save_5/RestoreV2:26*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
Ż
save_5/Assign_27Assignvf/dense/biassave_5/RestoreV2:27* 
_class
loc:@vf/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
´
save_5/Assign_28Assignvf/dense/bias/Adamsave_5/RestoreV2:28*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(* 
_class
loc:@vf/dense/bias
ś
save_5/Assign_29Assignvf/dense/bias/Adam_1save_5/RestoreV2:29*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(
ˇ
save_5/Assign_30Assignvf/dense/kernelsave_5/RestoreV2:30*
_output_shapes
:	<*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(
ź
save_5/Assign_31Assignvf/dense/kernel/Adamsave_5/RestoreV2:31*
use_locking(*
_output_shapes
:	<*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel
ž
save_5/Assign_32Assignvf/dense/kernel/Adam_1save_5/RestoreV2:32*
use_locking(*
_output_shapes
:	<*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel
ł
save_5/Assign_33Assignvf/dense_1/biassave_5/RestoreV2:33*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
¸
save_5/Assign_34Assignvf/dense_1/bias/Adamsave_5/RestoreV2:34*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(
ş
save_5/Assign_35Assignvf/dense_1/bias/Adam_1save_5/RestoreV2:35*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(
ź
save_5/Assign_36Assignvf/dense_1/kernelsave_5/RestoreV2:36*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
T0*
validate_shape(
Á
save_5/Assign_37Assignvf/dense_1/kernel/Adamsave_5/RestoreV2:37*
T0*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
validate_shape(
Ă
save_5/Assign_38Assignvf/dense_1/kernel/Adam_1save_5/RestoreV2:38*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(
˛
save_5/Assign_39Assignvf/dense_2/biassave_5/RestoreV2:39*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:
ˇ
save_5/Assign_40Assignvf/dense_2/bias/Adamsave_5/RestoreV2:40*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
š
save_5/Assign_41Assignvf/dense_2/bias/Adam_1save_5/RestoreV2:41*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias
ť
save_5/Assign_42Assignvf/dense_2/kernelsave_5/RestoreV2:42*
T0*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(
Ŕ
save_5/Assign_43Assignvf/dense_2/kernel/Adamsave_5/RestoreV2:43*
_output_shapes
:	*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel
Â
save_5/Assign_44Assignvf/dense_2/kernel/Adam_1save_5/RestoreV2:44*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(
ç
save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_2^save_5/Assign_20^save_5/Assign_21^save_5/Assign_22^save_5/Assign_23^save_5/Assign_24^save_5/Assign_25^save_5/Assign_26^save_5/Assign_27^save_5/Assign_28^save_5/Assign_29^save_5/Assign_3^save_5/Assign_30^save_5/Assign_31^save_5/Assign_32^save_5/Assign_33^save_5/Assign_34^save_5/Assign_35^save_5/Assign_36^save_5/Assign_37^save_5/Assign_38^save_5/Assign_39^save_5/Assign_4^save_5/Assign_40^save_5/Assign_41^save_5/Assign_42^save_5/Assign_43^save_5/Assign_44^save_5/Assign_5^save_5/Assign_6^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard
[
save_6/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
dtype0*
shape: *
_output_shapes
: 

save_6/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_fb7c283b122044759e849d1ffb559480/part
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_6/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_6/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
ň
save_6/SaveV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
ż
save_6/SaveV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ś
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*
_output_shapes
: *)
_class
loc:@save_6/ShardedFilename*
T0
Ł
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
N*

axis *
_output_shapes
:*
T0

save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(

save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
T0*
_output_shapes
: 
ő
save_6/RestoreV2/tensor_namesConst*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-
Â
!save_6/RestoreV2/shape_and_slicesConst*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-
÷
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
˘
save_6/AssignAssignbeta1_powersave_6/RestoreV2*
use_locking(*
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0
Ś
save_6/Assign_1Assignbeta2_powersave_6/RestoreV2:1* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
_output_shapes
: *
validate_shape(
­
save_6/Assign_2Assignpi/dense/biassave_6/RestoreV2:2*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@pi/dense/bias
ľ
save_6/Assign_3Assignpi/dense/kernelsave_6/RestoreV2:3*
_output_shapes
:	<*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel
ą
save_6/Assign_4Assignpi/dense_1/biassave_6/RestoreV2:4*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
ş
save_6/Assign_5Assignpi/dense_1/kernelsave_6/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
°
save_6/Assign_6Assignpi/dense_2/biassave_6/RestoreV2:6*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0
š
save_6/Assign_7Assignpi/dense_2/kernelsave_6/RestoreV2:7*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
Ś
save_6/Assign_8Assign
pi/log_stdsave_6/RestoreV2:8*
T0*
validate_shape(*
_class
loc:@pi/log_std*
_output_shapes
:*
use_locking(
­
save_6/Assign_9Assignvc/dense/biassave_6/RestoreV2:9* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
´
save_6/Assign_10Assignvc/dense/bias/Adamsave_6/RestoreV2:10*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:
ś
save_6/Assign_11Assignvc/dense/bias/Adam_1save_6/RestoreV2:11*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
_output_shapes	
:
ˇ
save_6/Assign_12Assignvc/dense/kernelsave_6/RestoreV2:12*
use_locking(*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
T0
ź
save_6/Assign_13Assignvc/dense/kernel/Adamsave_6/RestoreV2:13*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
validate_shape(*
use_locking(*
T0
ž
save_6/Assign_14Assignvc/dense/kernel/Adam_1save_6/RestoreV2:14*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	<*
use_locking(
ł
save_6/Assign_15Assignvc/dense_1/biassave_6/RestoreV2:15*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:*
T0
¸
save_6/Assign_16Assignvc/dense_1/bias/Adamsave_6/RestoreV2:16*
_output_shapes	
:*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(
ş
save_6/Assign_17Assignvc/dense_1/bias/Adam_1save_6/RestoreV2:17*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
use_locking(*
validate_shape(*
T0
ź
save_6/Assign_18Assignvc/dense_1/kernelsave_6/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:
*
T0
Á
save_6/Assign_19Assignvc/dense_1/kernel/Adamsave_6/RestoreV2:19*
use_locking(*
T0* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
Ă
save_6/Assign_20Assignvc/dense_1/kernel/Adam_1save_6/RestoreV2:20*
use_locking(*
T0* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
˛
save_6/Assign_21Assignvc/dense_2/biassave_6/RestoreV2:21*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
T0*
use_locking(
ˇ
save_6/Assign_22Assignvc/dense_2/bias/Adamsave_6/RestoreV2:22*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
š
save_6/Assign_23Assignvc/dense_2/bias/Adam_1save_6/RestoreV2:23*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
T0*
use_locking(
ť
save_6/Assign_24Assignvc/dense_2/kernelsave_6/RestoreV2:24*$
_class
loc:@vc/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	*
use_locking(
Ŕ
save_6/Assign_25Assignvc/dense_2/kernel/Adamsave_6/RestoreV2:25*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(
Â
save_6/Assign_26Assignvc/dense_2/kernel/Adam_1save_6/RestoreV2:26*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	
Ż
save_6/Assign_27Assignvf/dense/biassave_6/RestoreV2:27*
T0* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:
´
save_6/Assign_28Assignvf/dense/bias/Adamsave_6/RestoreV2:28*
_output_shapes	
:*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias*
T0
ś
save_6/Assign_29Assignvf/dense/bias/Adam_1save_6/RestoreV2:29* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0
ˇ
save_6/Assign_30Assignvf/dense/kernelsave_6/RestoreV2:30*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	<
ź
save_6/Assign_31Assignvf/dense/kernel/Adamsave_6/RestoreV2:31*
T0*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(
ž
save_6/Assign_32Assignvf/dense/kernel/Adam_1save_6/RestoreV2:32*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	<*
validate_shape(
ł
save_6/Assign_33Assignvf/dense_1/biassave_6/RestoreV2:33*
_output_shapes	
:*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0
¸
save_6/Assign_34Assignvf/dense_1/bias/Adamsave_6/RestoreV2:34*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:*
use_locking(
ş
save_6/Assign_35Assignvf/dense_1/bias/Adam_1save_6/RestoreV2:35*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(
ź
save_6/Assign_36Assignvf/dense_1/kernelsave_6/RestoreV2:36*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(
Á
save_6/Assign_37Assignvf/dense_1/kernel/Adamsave_6/RestoreV2:37*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0
Ă
save_6/Assign_38Assignvf/dense_1/kernel/Adam_1save_6/RestoreV2:38*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
˛
save_6/Assign_39Assignvf/dense_2/biassave_6/RestoreV2:39*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:
ˇ
save_6/Assign_40Assignvf/dense_2/bias/Adamsave_6/RestoreV2:40*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*"
_class
loc:@vf/dense_2/bias
š
save_6/Assign_41Assignvf/dense_2/bias/Adam_1save_6/RestoreV2:41*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(
ť
save_6/Assign_42Assignvf/dense_2/kernelsave_6/RestoreV2:42*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
T0*
use_locking(
Ŕ
save_6/Assign_43Assignvf/dense_2/kernel/Adamsave_6/RestoreV2:43*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
Â
save_6/Assign_44Assignvf/dense_2/kernel/Adam_1save_6/RestoreV2:44*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
ç
save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_2^save_6/Assign_20^save_6/Assign_21^save_6/Assign_22^save_6/Assign_23^save_6/Assign_24^save_6/Assign_25^save_6/Assign_26^save_6/Assign_27^save_6/Assign_28^save_6/Assign_29^save_6/Assign_3^save_6/Assign_30^save_6/Assign_31^save_6/Assign_32^save_6/Assign_33^save_6/Assign_34^save_6/Assign_35^save_6/Assign_36^save_6/Assign_37^save_6/Assign_38^save_6/Assign_39^save_6/Assign_4^save_6/Assign_40^save_6/Assign_41^save_6/Assign_42^save_6/Assign_43^save_6/Assign_44^save_6/Assign_5^save_6/Assign_6^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard
[
save_7/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_7/filenamePlaceholderWithDefaultsave_7/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_7/ConstPlaceholderWithDefaultsave_7/filename*
shape: *
dtype0*
_output_shapes
: 

save_7/StringJoin/inputs_1Const*<
value3B1 B+_temp_50fc4753ec71456cb3c0981f958568ec/part*
dtype0*
_output_shapes
: 
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
S
save_7/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_7/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 

save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
ň
save_7/SaveV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
ż
save_7/SaveV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ś
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*)
_class
loc:@save_7/ShardedFilename*
_output_shapes
: *
T0
Ł
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*
_output_shapes
:*
T0*
N*

axis 

save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(

save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency*
T0*
_output_shapes
: 
ő
save_7/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
Â
!save_7/RestoreV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-*
dtype0
÷
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
˘
save_7/AssignAssignbeta1_powersave_7/RestoreV2*
validate_shape(*
use_locking(*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias
Ś
save_7/Assign_1Assignbeta2_powersave_7/RestoreV2:1*
_output_shapes
: *
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0
­
save_7/Assign_2Assignpi/dense/biassave_7/RestoreV2:2* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0
ľ
save_7/Assign_3Assignpi/dense/kernelsave_7/RestoreV2:3*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	<*
use_locking(*
validate_shape(
ą
save_7/Assign_4Assignpi/dense_1/biassave_7/RestoreV2:4*
_output_shapes	
:*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0
ş
save_7/Assign_5Assignpi/dense_1/kernelsave_7/RestoreV2:5*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
validate_shape(
°
save_7/Assign_6Assignpi/dense_2/biassave_7/RestoreV2:6*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0
š
save_7/Assign_7Assignpi/dense_2/kernelsave_7/RestoreV2:7*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
T0
Ś
save_7/Assign_8Assign
pi/log_stdsave_7/RestoreV2:8*
_class
loc:@pi/log_std*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
­
save_7/Assign_9Assignvc/dense/biassave_7/RestoreV2:9*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
´
save_7/Assign_10Assignvc/dense/bias/Adamsave_7/RestoreV2:10*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:
ś
save_7/Assign_11Assignvc/dense/bias/Adam_1save_7/RestoreV2:11*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
validate_shape(*
T0
ˇ
save_7/Assign_12Assignvc/dense/kernelsave_7/RestoreV2:12*
validate_shape(*
T0*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vc/dense/kernel
ź
save_7/Assign_13Assignvc/dense/kernel/Adamsave_7/RestoreV2:13*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	<*
T0*
use_locking(
ž
save_7/Assign_14Assignvc/dense/kernel/Adam_1save_7/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	<
ł
save_7/Assign_15Assignvc/dense_1/biassave_7/RestoreV2:15*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(
¸
save_7/Assign_16Assignvc/dense_1/bias/Adamsave_7/RestoreV2:16*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
ş
save_7/Assign_17Assignvc/dense_1/bias/Adam_1save_7/RestoreV2:17*
use_locking(*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(
ź
save_7/Assign_18Assignvc/dense_1/kernelsave_7/RestoreV2:18*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:

Á
save_7/Assign_19Assignvc/dense_1/kernel/Adamsave_7/RestoreV2:19*
T0*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
Ă
save_7/Assign_20Assignvc/dense_1/kernel/Adam_1save_7/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:

˛
save_7/Assign_21Assignvc/dense_2/biassave_7/RestoreV2:21*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0
ˇ
save_7/Assign_22Assignvc/dense_2/bias/Adamsave_7/RestoreV2:22*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
š
save_7/Assign_23Assignvc/dense_2/bias/Adam_1save_7/RestoreV2:23*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:
ť
save_7/Assign_24Assignvc/dense_2/kernelsave_7/RestoreV2:24*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
Ŕ
save_7/Assign_25Assignvc/dense_2/kernel/Adamsave_7/RestoreV2:25*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	
Â
save_7/Assign_26Assignvc/dense_2/kernel/Adam_1save_7/RestoreV2:26*
_output_shapes
:	*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
Ż
save_7/Assign_27Assignvf/dense/biassave_7/RestoreV2:27*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(* 
_class
loc:@vf/dense/bias
´
save_7/Assign_28Assignvf/dense/bias/Adamsave_7/RestoreV2:28*
_output_shapes	
:*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(*
T0
ś
save_7/Assign_29Assignvf/dense/bias/Adam_1save_7/RestoreV2:29*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
use_locking(
ˇ
save_7/Assign_30Assignvf/dense/kernelsave_7/RestoreV2:30*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel
ź
save_7/Assign_31Assignvf/dense/kernel/Adamsave_7/RestoreV2:31*
_output_shapes
:	<*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(
ž
save_7/Assign_32Assignvf/dense/kernel/Adam_1save_7/RestoreV2:32*
T0*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(
ł
save_7/Assign_33Assignvf/dense_1/biassave_7/RestoreV2:33*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:
¸
save_7/Assign_34Assignvf/dense_1/bias/Adamsave_7/RestoreV2:34*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(*
T0
ş
save_7/Assign_35Assignvf/dense_1/bias/Adam_1save_7/RestoreV2:35*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(*
T0
ź
save_7/Assign_36Assignvf/dense_1/kernelsave_7/RestoreV2:36* 
_output_shapes
:
*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel
Á
save_7/Assign_37Assignvf/dense_1/kernel/Adamsave_7/RestoreV2:37*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel
Ă
save_7/Assign_38Assignvf/dense_1/kernel/Adam_1save_7/RestoreV2:38* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel
˛
save_7/Assign_39Assignvf/dense_2/biassave_7/RestoreV2:39*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0
ˇ
save_7/Assign_40Assignvf/dense_2/bias/Adamsave_7/RestoreV2:40*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(
š
save_7/Assign_41Assignvf/dense_2/bias/Adam_1save_7/RestoreV2:41*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias
ť
save_7/Assign_42Assignvf/dense_2/kernelsave_7/RestoreV2:42*
T0*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel
Ŕ
save_7/Assign_43Assignvf/dense_2/kernel/Adamsave_7/RestoreV2:43*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
Â
save_7/Assign_44Assignvf/dense_2/kernel/Adam_1save_7/RestoreV2:44*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	
ç
save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_2^save_7/Assign_20^save_7/Assign_21^save_7/Assign_22^save_7/Assign_23^save_7/Assign_24^save_7/Assign_25^save_7/Assign_26^save_7/Assign_27^save_7/Assign_28^save_7/Assign_29^save_7/Assign_3^save_7/Assign_30^save_7/Assign_31^save_7/Assign_32^save_7/Assign_33^save_7/Assign_34^save_7/Assign_35^save_7/Assign_36^save_7/Assign_37^save_7/Assign_38^save_7/Assign_39^save_7/Assign_4^save_7/Assign_40^save_7/Assign_41^save_7/Assign_42^save_7/Assign_43^save_7/Assign_44^save_7/Assign_5^save_7/Assign_6^save_7/Assign_7^save_7/Assign_8^save_7/Assign_9
1
save_7/restore_allNoOp^save_7/restore_shard
[
save_8/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_8/filenamePlaceholderWithDefaultsave_8/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_8/ConstPlaceholderWithDefaultsave_8/filename*
shape: *
_output_shapes
: *
dtype0

save_8/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_fc04704043784c24ab7b80fe0d6fdff7/part*
_output_shapes
: 
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_8/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_8/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
ň
save_8/SaveV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
ż
save_8/SaveV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ś
save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*
_output_shapes
: *)
_class
loc:@save_8/ShardedFilename*
T0
Ł
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilename^save_8/control_dependency*
N*
T0*

axis *
_output_shapes
:

save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(

save_8/IdentityIdentitysave_8/Const^save_8/MergeV2Checkpoints^save_8/control_dependency*
T0*
_output_shapes
: 
ő
save_8/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
Â
!save_8/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
÷
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
˘
save_8/AssignAssignbeta1_powersave_8/RestoreV2*
use_locking(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0*
validate_shape(
Ś
save_8/Assign_1Assignbeta2_powersave_8/RestoreV2:1*
_output_shapes
: *
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0
­
save_8/Assign_2Assignpi/dense/biassave_8/RestoreV2:2*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
use_locking(
ľ
save_8/Assign_3Assignpi/dense/kernelsave_8/RestoreV2:3*
validate_shape(*
use_locking(*
_output_shapes
:	<*
T0*"
_class
loc:@pi/dense/kernel
ą
save_8/Assign_4Assignpi/dense_1/biassave_8/RestoreV2:4*
_output_shapes	
:*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(
ş
save_8/Assign_5Assignpi/dense_1/kernelsave_8/RestoreV2:5*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(
°
save_8/Assign_6Assignpi/dense_2/biassave_8/RestoreV2:6*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0
š
save_8/Assign_7Assignpi/dense_2/kernelsave_8/RestoreV2:7*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
Ś
save_8/Assign_8Assign
pi/log_stdsave_8/RestoreV2:8*
validate_shape(*
_class
loc:@pi/log_std*
T0*
use_locking(*
_output_shapes
:
­
save_8/Assign_9Assignvc/dense/biassave_8/RestoreV2:9*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
use_locking(
´
save_8/Assign_10Assignvc/dense/bias/Adamsave_8/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(
ś
save_8/Assign_11Assignvc/dense/bias/Adam_1save_8/RestoreV2:11*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
ˇ
save_8/Assign_12Assignvc/dense/kernelsave_8/RestoreV2:12*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<*
validate_shape(*
use_locking(
ź
save_8/Assign_13Assignvc/dense/kernel/Adamsave_8/RestoreV2:13*
validate_shape(*
T0*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vc/dense/kernel
ž
save_8/Assign_14Assignvc/dense/kernel/Adam_1save_8/RestoreV2:14*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
validate_shape(
ł
save_8/Assign_15Assignvc/dense_1/biassave_8/RestoreV2:15*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
¸
save_8/Assign_16Assignvc/dense_1/bias/Adamsave_8/RestoreV2:16*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
ş
save_8/Assign_17Assignvc/dense_1/bias/Adam_1save_8/RestoreV2:17*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:
ź
save_8/Assign_18Assignvc/dense_1/kernelsave_8/RestoreV2:18* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0
Á
save_8/Assign_19Assignvc/dense_1/kernel/Adamsave_8/RestoreV2:19*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
Ă
save_8/Assign_20Assignvc/dense_1/kernel/Adam_1save_8/RestoreV2:20*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
T0
˛
save_8/Assign_21Assignvc/dense_2/biassave_8/RestoreV2:21*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
ˇ
save_8/Assign_22Assignvc/dense_2/bias/Adamsave_8/RestoreV2:22*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
š
save_8/Assign_23Assignvc/dense_2/bias/Adam_1save_8/RestoreV2:23*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(
ť
save_8/Assign_24Assignvc/dense_2/kernelsave_8/RestoreV2:24*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
use_locking(*
validate_shape(*
T0
Ŕ
save_8/Assign_25Assignvc/dense_2/kernel/Adamsave_8/RestoreV2:25*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	
Â
save_8/Assign_26Assignvc/dense_2/kernel/Adam_1save_8/RestoreV2:26*
T0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(
Ż
save_8/Assign_27Assignvf/dense/biassave_8/RestoreV2:27*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
use_locking(
´
save_8/Assign_28Assignvf/dense/bias/Adamsave_8/RestoreV2:28*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(* 
_class
loc:@vf/dense/bias
ś
save_8/Assign_29Assignvf/dense/bias/Adam_1save_8/RestoreV2:29* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(
ˇ
save_8/Assign_30Assignvf/dense/kernelsave_8/RestoreV2:30*
use_locking(*
validate_shape(*
_output_shapes
:	<*
T0*"
_class
loc:@vf/dense/kernel
ź
save_8/Assign_31Assignvf/dense/kernel/Adamsave_8/RestoreV2:31*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	<*
T0
ž
save_8/Assign_32Assignvf/dense/kernel/Adam_1save_8/RestoreV2:32*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel
ł
save_8/Assign_33Assignvf/dense_1/biassave_8/RestoreV2:33*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
¸
save_8/Assign_34Assignvf/dense_1/bias/Adamsave_8/RestoreV2:34*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
use_locking(
ş
save_8/Assign_35Assignvf/dense_1/bias/Adam_1save_8/RestoreV2:35*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
ź
save_8/Assign_36Assignvf/dense_1/kernelsave_8/RestoreV2:36*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:

Á
save_8/Assign_37Assignvf/dense_1/kernel/Adamsave_8/RestoreV2:37* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel
Ă
save_8/Assign_38Assignvf/dense_1/kernel/Adam_1save_8/RestoreV2:38*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:
*
validate_shape(
˛
save_8/Assign_39Assignvf/dense_2/biassave_8/RestoreV2:39*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
ˇ
save_8/Assign_40Assignvf/dense_2/bias/Adamsave_8/RestoreV2:40*
_output_shapes
:*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(*
T0
š
save_8/Assign_41Assignvf/dense_2/bias/Adam_1save_8/RestoreV2:41*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(*
T0*
_output_shapes
:
ť
save_8/Assign_42Assignvf/dense_2/kernelsave_8/RestoreV2:42*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	
Ŕ
save_8/Assign_43Assignvf/dense_2/kernel/Adamsave_8/RestoreV2:43*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0*
use_locking(
Â
save_8/Assign_44Assignvf/dense_2/kernel/Adam_1save_8/RestoreV2:44*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0*
use_locking(
ç
save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_16^save_8/Assign_17^save_8/Assign_18^save_8/Assign_19^save_8/Assign_2^save_8/Assign_20^save_8/Assign_21^save_8/Assign_22^save_8/Assign_23^save_8/Assign_24^save_8/Assign_25^save_8/Assign_26^save_8/Assign_27^save_8/Assign_28^save_8/Assign_29^save_8/Assign_3^save_8/Assign_30^save_8/Assign_31^save_8/Assign_32^save_8/Assign_33^save_8/Assign_34^save_8/Assign_35^save_8/Assign_36^save_8/Assign_37^save_8/Assign_38^save_8/Assign_39^save_8/Assign_4^save_8/Assign_40^save_8/Assign_41^save_8/Assign_42^save_8/Assign_43^save_8/Assign_44^save_8/Assign_5^save_8/Assign_6^save_8/Assign_7^save_8/Assign_8^save_8/Assign_9
1
save_8/restore_allNoOp^save_8/restore_shard
[
save_9/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_9/filenamePlaceholderWithDefaultsave_9/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_9/ConstPlaceholderWithDefaultsave_9/filename*
_output_shapes
: *
dtype0*
shape: 

save_9/StringJoin/inputs_1Const*<
value3B1 B+_temp_40fa7f60d8994ac58b1d30a6da048ec1/part*
_output_shapes
: *
dtype0
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
S
save_9/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_9/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
ň
save_9/SaveV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
ż
save_9/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ś
save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_9/ShardedFilename
Ł
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilename^save_9/control_dependency*

axis *
T0*
N*
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
ő
save_9/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
Â
!save_9/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
÷
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
˘
save_9/AssignAssignbeta1_powersave_9/RestoreV2* 
_class
loc:@vc/dense/bias*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
Ś
save_9/Assign_1Assignbeta2_powersave_9/RestoreV2:1*
use_locking(*
T0*
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias
­
save_9/Assign_2Assignpi/dense/biassave_9/RestoreV2:2*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
validate_shape(
ľ
save_9/Assign_3Assignpi/dense/kernelsave_9/RestoreV2:3*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	<
ą
save_9/Assign_4Assignpi/dense_1/biassave_9/RestoreV2:4*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
use_locking(*
T0
ş
save_9/Assign_5Assignpi/dense_1/kernelsave_9/RestoreV2:5*
validate_shape(*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
T0
°
save_9/Assign_6Assignpi/dense_2/biassave_9/RestoreV2:6*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0*
_output_shapes
:
š
save_9/Assign_7Assignpi/dense_2/kernelsave_9/RestoreV2:7*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
:	*
validate_shape(*
use_locking(
Ś
save_9/Assign_8Assign
pi/log_stdsave_9/RestoreV2:8*
_class
loc:@pi/log_std*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
­
save_9/Assign_9Assignvc/dense/biassave_9/RestoreV2:9* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(
´
save_9/Assign_10Assignvc/dense/bias/Adamsave_9/RestoreV2:10*
use_locking(*
_output_shapes	
:*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0
ś
save_9/Assign_11Assignvc/dense/bias/Adam_1save_9/RestoreV2:11*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias
ˇ
save_9/Assign_12Assignvc/dense/kernelsave_9/RestoreV2:12*
use_locking(*
_output_shapes
:	<*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel
ź
save_9/Assign_13Assignvc/dense/kernel/Adamsave_9/RestoreV2:13*
validate_shape(*
T0*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vc/dense/kernel
ž
save_9/Assign_14Assignvc/dense/kernel/Adam_1save_9/RestoreV2:14*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0
ł
save_9/Assign_15Assignvc/dense_1/biassave_9/RestoreV2:15*
_output_shapes	
:*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(
¸
save_9/Assign_16Assignvc/dense_1/bias/Adamsave_9/RestoreV2:16*"
_class
loc:@vc/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(
ş
save_9/Assign_17Assignvc/dense_1/bias/Adam_1save_9/RestoreV2:17*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
use_locking(*
T0
ź
save_9/Assign_18Assignvc/dense_1/kernelsave_9/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:

Á
save_9/Assign_19Assignvc/dense_1/kernel/Adamsave_9/RestoreV2:19*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ă
save_9/Assign_20Assignvc/dense_1/kernel/Adam_1save_9/RestoreV2:20* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(
˛
save_9/Assign_21Assignvc/dense_2/biassave_9/RestoreV2:21*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:
ˇ
save_9/Assign_22Assignvc/dense_2/bias/Adamsave_9/RestoreV2:22*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0
š
save_9/Assign_23Assignvc/dense_2/bias/Adam_1save_9/RestoreV2:23*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0
ť
save_9/Assign_24Assignvc/dense_2/kernelsave_9/RestoreV2:24*
validate_shape(*
T0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
use_locking(
Ŕ
save_9/Assign_25Assignvc/dense_2/kernel/Adamsave_9/RestoreV2:25*
use_locking(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0
Â
save_9/Assign_26Assignvc/dense_2/kernel/Adam_1save_9/RestoreV2:26*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
Ż
save_9/Assign_27Assignvf/dense/biassave_9/RestoreV2:27*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(* 
_class
loc:@vf/dense/bias
´
save_9/Assign_28Assignvf/dense/bias/Adamsave_9/RestoreV2:28*
T0*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
ś
save_9/Assign_29Assignvf/dense/bias/Adam_1save_9/RestoreV2:29*
T0* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:
ˇ
save_9/Assign_30Assignvf/dense/kernelsave_9/RestoreV2:30*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	<
ź
save_9/Assign_31Assignvf/dense/kernel/Adamsave_9/RestoreV2:31*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(
ž
save_9/Assign_32Assignvf/dense/kernel/Adam_1save_9/RestoreV2:32*
T0*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
use_locking(
ł
save_9/Assign_33Assignvf/dense_1/biassave_9/RestoreV2:33*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:
¸
save_9/Assign_34Assignvf/dense_1/bias/Adamsave_9/RestoreV2:34*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(*
T0
ş
save_9/Assign_35Assignvf/dense_1/bias/Adam_1save_9/RestoreV2:35*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(
ź
save_9/Assign_36Assignvf/dense_1/kernelsave_9/RestoreV2:36* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(*
validate_shape(
Á
save_9/Assign_37Assignvf/dense_1/kernel/Adamsave_9/RestoreV2:37* 
_output_shapes
:
*
T0*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(
Ă
save_9/Assign_38Assignvf/dense_1/kernel/Adam_1save_9/RestoreV2:38*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:

˛
save_9/Assign_39Assignvf/dense_2/biassave_9/RestoreV2:39*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
ˇ
save_9/Assign_40Assignvf/dense_2/bias/Adamsave_9/RestoreV2:40*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:
š
save_9/Assign_41Assignvf/dense_2/bias/Adam_1save_9/RestoreV2:41*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
ť
save_9/Assign_42Assignvf/dense_2/kernelsave_9/RestoreV2:42*$
_class
loc:@vf/dense_2/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	
Ŕ
save_9/Assign_43Assignvf/dense_2/kernel/Adamsave_9/RestoreV2:43*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_9/Assign_44Assignvf/dense_2/kernel/Adam_1save_9/RestoreV2:44*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
ç
save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_16^save_9/Assign_17^save_9/Assign_18^save_9/Assign_19^save_9/Assign_2^save_9/Assign_20^save_9/Assign_21^save_9/Assign_22^save_9/Assign_23^save_9/Assign_24^save_9/Assign_25^save_9/Assign_26^save_9/Assign_27^save_9/Assign_28^save_9/Assign_29^save_9/Assign_3^save_9/Assign_30^save_9/Assign_31^save_9/Assign_32^save_9/Assign_33^save_9/Assign_34^save_9/Assign_35^save_9/Assign_36^save_9/Assign_37^save_9/Assign_38^save_9/Assign_39^save_9/Assign_4^save_9/Assign_40^save_9/Assign_41^save_9/Assign_42^save_9/Assign_43^save_9/Assign_44^save_9/Assign_5^save_9/Assign_6^save_9/Assign_7^save_9/Assign_8^save_9/Assign_9
1
save_9/restore_allNoOp^save_9/restore_shard
\
save_10/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
t
save_10/filenamePlaceholderWithDefaultsave_10/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_10/ConstPlaceholderWithDefaultsave_10/filename*
dtype0*
shape: *
_output_shapes
: 

save_10/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_f4e1b4bf2d1143288a23e54c813dde8f/part
~
save_10/StringJoin
StringJoinsave_10/Constsave_10/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_10/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_10/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_10/ShardedFilenameShardedFilenamesave_10/StringJoinsave_10/ShardedFilename/shardsave_10/num_shards*
_output_shapes
: 
ó
save_10/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ŕ
save_10/SaveV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ş
save_10/SaveV2SaveV2save_10/ShardedFilenamesave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_10/control_dependencyIdentitysave_10/ShardedFilename^save_10/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_10/ShardedFilename
Ś
.save_10/MergeV2Checkpoints/checkpoint_prefixesPacksave_10/ShardedFilename^save_10/control_dependency*
_output_shapes
:*
T0*
N*

axis 

save_10/MergeV2CheckpointsMergeV2Checkpoints.save_10/MergeV2Checkpoints/checkpoint_prefixessave_10/Const*
delete_old_dirs(

save_10/IdentityIdentitysave_10/Const^save_10/MergeV2Checkpoints^save_10/control_dependency*
_output_shapes
: *
T0
ö
save_10/RestoreV2/tensor_namesConst*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
Ă
"save_10/RestoreV2/shape_and_slicesConst*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-
ű
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
¤
save_10/AssignAssignbeta1_powersave_10/RestoreV2*
T0* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: 
¨
save_10/Assign_1Assignbeta2_powersave_10/RestoreV2:1*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0*
use_locking(
Ż
save_10/Assign_2Assignpi/dense/biassave_10/RestoreV2:2*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@pi/dense/bias*
T0
ˇ
save_10/Assign_3Assignpi/dense/kernelsave_10/RestoreV2:3*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	<*
use_locking(
ł
save_10/Assign_4Assignpi/dense_1/biassave_10/RestoreV2:4*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
ź
save_10/Assign_5Assignpi/dense_1/kernelsave_10/RestoreV2:5*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:
*
validate_shape(
˛
save_10/Assign_6Assignpi/dense_2/biassave_10/RestoreV2:6*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias
ť
save_10/Assign_7Assignpi/dense_2/kernelsave_10/RestoreV2:7*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
validate_shape(
¨
save_10/Assign_8Assign
pi/log_stdsave_10/RestoreV2:8*
_output_shapes
:*
validate_shape(*
use_locking(*
_class
loc:@pi/log_std*
T0
Ż
save_10/Assign_9Assignvc/dense/biassave_10/RestoreV2:9*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
T0
ś
save_10/Assign_10Assignvc/dense/bias/Adamsave_10/RestoreV2:10* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(
¸
save_10/Assign_11Assignvc/dense/bias/Adam_1save_10/RestoreV2:11*
use_locking(*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:
š
save_10/Assign_12Assignvc/dense/kernelsave_10/RestoreV2:12*
use_locking(*
_output_shapes
:	<*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(
ž
save_10/Assign_13Assignvc/dense/kernel/Adamsave_10/RestoreV2:13*
T0*
validate_shape(*"
_class
loc:@vc/dense/kernel*
use_locking(*
_output_shapes
:	<
Ŕ
save_10/Assign_14Assignvc/dense/kernel/Adam_1save_10/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
validate_shape(*
T0*
use_locking(
ľ
save_10/Assign_15Assignvc/dense_1/biassave_10/RestoreV2:15*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
use_locking(*
validate_shape(
ş
save_10/Assign_16Assignvc/dense_1/bias/Adamsave_10/RestoreV2:16*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
ź
save_10/Assign_17Assignvc/dense_1/bias/Adam_1save_10/RestoreV2:17*
use_locking(*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:*
T0
ž
save_10/Assign_18Assignvc/dense_1/kernelsave_10/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:

Ă
save_10/Assign_19Assignvc/dense_1/kernel/Adamsave_10/RestoreV2:19*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:

Ĺ
save_10/Assign_20Assignvc/dense_1/kernel/Adam_1save_10/RestoreV2:20* 
_output_shapes
:
*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
´
save_10/Assign_21Assignvc/dense_2/biassave_10/RestoreV2:21*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(
š
save_10/Assign_22Assignvc/dense_2/bias/Adamsave_10/RestoreV2:22*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0
ť
save_10/Assign_23Assignvc/dense_2/bias/Adam_1save_10/RestoreV2:23*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
˝
save_10/Assign_24Assignvc/dense_2/kernelsave_10/RestoreV2:24*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(
Â
save_10/Assign_25Assignvc/dense_2/kernel/Adamsave_10/RestoreV2:25*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0
Ä
save_10/Assign_26Assignvc/dense_2/kernel/Adam_1save_10/RestoreV2:26*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(
ą
save_10/Assign_27Assignvf/dense/biassave_10/RestoreV2:27*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
use_locking(
ś
save_10/Assign_28Assignvf/dense/bias/Adamsave_10/RestoreV2:28*
T0*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
validate_shape(
¸
save_10/Assign_29Assignvf/dense/bias/Adam_1save_10/RestoreV2:29*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
š
save_10/Assign_30Assignvf/dense/kernelsave_10/RestoreV2:30*
T0*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(
ž
save_10/Assign_31Assignvf/dense/kernel/Adamsave_10/RestoreV2:31*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
T0*
validate_shape(
Ŕ
save_10/Assign_32Assignvf/dense/kernel/Adam_1save_10/RestoreV2:32*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(
ľ
save_10/Assign_33Assignvf/dense_1/biassave_10/RestoreV2:33*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
ş
save_10/Assign_34Assignvf/dense_1/bias/Adamsave_10/RestoreV2:34*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:
ź
save_10/Assign_35Assignvf/dense_1/bias/Adam_1save_10/RestoreV2:35*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
ž
save_10/Assign_36Assignvf/dense_1/kernelsave_10/RestoreV2:36* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(*
T0
Ă
save_10/Assign_37Assignvf/dense_1/kernel/Adamsave_10/RestoreV2:37*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:

Ĺ
save_10/Assign_38Assignvf/dense_1/kernel/Adam_1save_10/RestoreV2:38*
validate_shape(*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(
´
save_10/Assign_39Assignvf/dense_2/biassave_10/RestoreV2:39*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_10/Assign_40Assignvf/dense_2/bias/Adamsave_10/RestoreV2:40*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias
ť
save_10/Assign_41Assignvf/dense_2/bias/Adam_1save_10/RestoreV2:41*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(
˝
save_10/Assign_42Assignvf/dense_2/kernelsave_10/RestoreV2:42*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
Â
save_10/Assign_43Assignvf/dense_2/kernel/Adamsave_10/RestoreV2:43*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0*$
_class
loc:@vf/dense_2/kernel
Ä
save_10/Assign_44Assignvf/dense_2/kernel/Adam_1save_10/RestoreV2:44*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(*
validate_shape(*
T0

save_10/restore_shardNoOp^save_10/Assign^save_10/Assign_1^save_10/Assign_10^save_10/Assign_11^save_10/Assign_12^save_10/Assign_13^save_10/Assign_14^save_10/Assign_15^save_10/Assign_16^save_10/Assign_17^save_10/Assign_18^save_10/Assign_19^save_10/Assign_2^save_10/Assign_20^save_10/Assign_21^save_10/Assign_22^save_10/Assign_23^save_10/Assign_24^save_10/Assign_25^save_10/Assign_26^save_10/Assign_27^save_10/Assign_28^save_10/Assign_29^save_10/Assign_3^save_10/Assign_30^save_10/Assign_31^save_10/Assign_32^save_10/Assign_33^save_10/Assign_34^save_10/Assign_35^save_10/Assign_36^save_10/Assign_37^save_10/Assign_38^save_10/Assign_39^save_10/Assign_4^save_10/Assign_40^save_10/Assign_41^save_10/Assign_42^save_10/Assign_43^save_10/Assign_44^save_10/Assign_5^save_10/Assign_6^save_10/Assign_7^save_10/Assign_8^save_10/Assign_9
3
save_10/restore_allNoOp^save_10/restore_shard
\
save_11/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_11/filenamePlaceholderWithDefaultsave_11/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_11/ConstPlaceholderWithDefaultsave_11/filename*
dtype0*
shape: *
_output_shapes
: 

save_11/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d8cbb5f306c0459fb074e801763a8a08/part
~
save_11/StringJoin
StringJoinsave_11/Constsave_11/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_11/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
_
save_11/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_11/ShardedFilenameShardedFilenamesave_11/StringJoinsave_11/ShardedFilename/shardsave_11/num_shards*
_output_shapes
: 
ó
save_11/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ŕ
save_11/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ş
save_11/SaveV2SaveV2save_11/ShardedFilenamesave_11/SaveV2/tensor_namessave_11/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_11/control_dependencyIdentitysave_11/ShardedFilename^save_11/SaveV2*
_output_shapes
: **
_class 
loc:@save_11/ShardedFilename*
T0
Ś
.save_11/MergeV2Checkpoints/checkpoint_prefixesPacksave_11/ShardedFilename^save_11/control_dependency*
_output_shapes
:*

axis *
T0*
N

save_11/MergeV2CheckpointsMergeV2Checkpoints.save_11/MergeV2Checkpoints/checkpoint_prefixessave_11/Const*
delete_old_dirs(

save_11/IdentityIdentitysave_11/Const^save_11/MergeV2Checkpoints^save_11/control_dependency*
_output_shapes
: *
T0
ö
save_11/RestoreV2/tensor_namesConst*
_output_shapes
:-*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ă
"save_11/RestoreV2/shape_and_slicesConst*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-
ű
save_11/RestoreV2	RestoreV2save_11/Constsave_11/RestoreV2/tensor_names"save_11/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
¤
save_11/AssignAssignbeta1_powersave_11/RestoreV2* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: *
validate_shape(*
use_locking(
¨
save_11/Assign_1Assignbeta2_powersave_11/RestoreV2:1*
_output_shapes
: *
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(
Ż
save_11/Assign_2Assignpi/dense/biassave_11/RestoreV2:2* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:
ˇ
save_11/Assign_3Assignpi/dense/kernelsave_11/RestoreV2:3*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	<
ł
save_11/Assign_4Assignpi/dense_1/biassave_11/RestoreV2:4*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
validate_shape(
ź
save_11/Assign_5Assignpi/dense_1/kernelsave_11/RestoreV2:5*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:
*
validate_shape(
˛
save_11/Assign_6Assignpi/dense_2/biassave_11/RestoreV2:6*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias
ť
save_11/Assign_7Assignpi/dense_2/kernelsave_11/RestoreV2:7*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
T0*
use_locking(
¨
save_11/Assign_8Assign
pi/log_stdsave_11/RestoreV2:8*
_class
loc:@pi/log_std*
_output_shapes
:*
T0*
use_locking(*
validate_shape(
Ż
save_11/Assign_9Assignvc/dense/biassave_11/RestoreV2:9*
use_locking(*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
validate_shape(
ś
save_11/Assign_10Assignvc/dense/bias/Adamsave_11/RestoreV2:10*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias
¸
save_11/Assign_11Assignvc/dense/bias/Adam_1save_11/RestoreV2:11*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(* 
_class
loc:@vc/dense/bias
š
save_11/Assign_12Assignvc/dense/kernelsave_11/RestoreV2:12*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	<
ž
save_11/Assign_13Assignvc/dense/kernel/Adamsave_11/RestoreV2:13*
T0*
use_locking(*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@vc/dense/kernel
Ŕ
save_11/Assign_14Assignvc/dense/kernel/Adam_1save_11/RestoreV2:14*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
validate_shape(*
use_locking(
ľ
save_11/Assign_15Assignvc/dense_1/biassave_11/RestoreV2:15*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
use_locking(*
T0*
validate_shape(
ş
save_11/Assign_16Assignvc/dense_1/bias/Adamsave_11/RestoreV2:16*
use_locking(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0*
validate_shape(
ź
save_11/Assign_17Assignvc/dense_1/bias/Adam_1save_11/RestoreV2:17*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
ž
save_11/Assign_18Assignvc/dense_1/kernelsave_11/RestoreV2:18*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:

Ă
save_11/Assign_19Assignvc/dense_1/kernel/Adamsave_11/RestoreV2:19*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
use_locking(
Ĺ
save_11/Assign_20Assignvc/dense_1/kernel/Adam_1save_11/RestoreV2:20* 
_output_shapes
:
*
T0*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
´
save_11/Assign_21Assignvc/dense_2/biassave_11/RestoreV2:21*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
š
save_11/Assign_22Assignvc/dense_2/bias/Adamsave_11/RestoreV2:22*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
ť
save_11/Assign_23Assignvc/dense_2/bias/Adam_1save_11/RestoreV2:23*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(
˝
save_11/Assign_24Assignvc/dense_2/kernelsave_11/RestoreV2:24*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0
Â
save_11/Assign_25Assignvc/dense_2/kernel/Adamsave_11/RestoreV2:25*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
Ä
save_11/Assign_26Assignvc/dense_2/kernel/Adam_1save_11/RestoreV2:26*
_output_shapes
:	*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vc/dense_2/kernel
ą
save_11/Assign_27Assignvf/dense/biassave_11/RestoreV2:27*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vf/dense/bias
ś
save_11/Assign_28Assignvf/dense/bias/Adamsave_11/RestoreV2:28*
use_locking(*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
validate_shape(
¸
save_11/Assign_29Assignvf/dense/bias/Adam_1save_11/RestoreV2:29*
T0* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:
š
save_11/Assign_30Assignvf/dense/kernelsave_11/RestoreV2:30*
_output_shapes
:	<*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel
ž
save_11/Assign_31Assignvf/dense/kernel/Adamsave_11/RestoreV2:31*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
T0*
validate_shape(
Ŕ
save_11/Assign_32Assignvf/dense/kernel/Adam_1save_11/RestoreV2:32*
use_locking(*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	<*
validate_shape(
ľ
save_11/Assign_33Assignvf/dense_1/biassave_11/RestoreV2:33*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
ş
save_11/Assign_34Assignvf/dense_1/bias/Adamsave_11/RestoreV2:34*
T0*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(
ź
save_11/Assign_35Assignvf/dense_1/bias/Adam_1save_11/RestoreV2:35*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0
ž
save_11/Assign_36Assignvf/dense_1/kernelsave_11/RestoreV2:36*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:

Ă
save_11/Assign_37Assignvf/dense_1/kernel/Adamsave_11/RestoreV2:37* 
_output_shapes
:
*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
T0
Ĺ
save_11/Assign_38Assignvf/dense_1/kernel/Adam_1save_11/RestoreV2:38*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:
*
validate_shape(
´
save_11/Assign_39Assignvf/dense_2/biassave_11/RestoreV2:39*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
š
save_11/Assign_40Assignvf/dense_2/bias/Adamsave_11/RestoreV2:40*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
ť
save_11/Assign_41Assignvf/dense_2/bias/Adam_1save_11/RestoreV2:41*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
˝
save_11/Assign_42Assignvf/dense_2/kernelsave_11/RestoreV2:42*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	
Â
save_11/Assign_43Assignvf/dense_2/kernel/Adamsave_11/RestoreV2:43*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
Ä
save_11/Assign_44Assignvf/dense_2/kernel/Adam_1save_11/RestoreV2:44*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0*
_output_shapes
:	*
validate_shape(

save_11/restore_shardNoOp^save_11/Assign^save_11/Assign_1^save_11/Assign_10^save_11/Assign_11^save_11/Assign_12^save_11/Assign_13^save_11/Assign_14^save_11/Assign_15^save_11/Assign_16^save_11/Assign_17^save_11/Assign_18^save_11/Assign_19^save_11/Assign_2^save_11/Assign_20^save_11/Assign_21^save_11/Assign_22^save_11/Assign_23^save_11/Assign_24^save_11/Assign_25^save_11/Assign_26^save_11/Assign_27^save_11/Assign_28^save_11/Assign_29^save_11/Assign_3^save_11/Assign_30^save_11/Assign_31^save_11/Assign_32^save_11/Assign_33^save_11/Assign_34^save_11/Assign_35^save_11/Assign_36^save_11/Assign_37^save_11/Assign_38^save_11/Assign_39^save_11/Assign_4^save_11/Assign_40^save_11/Assign_41^save_11/Assign_42^save_11/Assign_43^save_11/Assign_44^save_11/Assign_5^save_11/Assign_6^save_11/Assign_7^save_11/Assign_8^save_11/Assign_9
3
save_11/restore_allNoOp^save_11/restore_shard
\
save_12/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_12/filenamePlaceholderWithDefaultsave_12/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_12/ConstPlaceholderWithDefaultsave_12/filename*
shape: *
dtype0*
_output_shapes
: 

save_12/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_e26abd121e124ba6be72b4c1eb7977be/part
~
save_12/StringJoin
StringJoinsave_12/Constsave_12/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_12/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
_
save_12/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_12/ShardedFilenameShardedFilenamesave_12/StringJoinsave_12/ShardedFilename/shardsave_12/num_shards*
_output_shapes
: 
ó
save_12/SaveV2/tensor_namesConst*
_output_shapes
:-*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ŕ
save_12/SaveV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ş
save_12/SaveV2SaveV2save_12/ShardedFilenamesave_12/SaveV2/tensor_namessave_12/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_12/control_dependencyIdentitysave_12/ShardedFilename^save_12/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_12/ShardedFilename
Ś
.save_12/MergeV2Checkpoints/checkpoint_prefixesPacksave_12/ShardedFilename^save_12/control_dependency*
T0*
_output_shapes
:*
N*

axis 

save_12/MergeV2CheckpointsMergeV2Checkpoints.save_12/MergeV2Checkpoints/checkpoint_prefixessave_12/Const*
delete_old_dirs(

save_12/IdentityIdentitysave_12/Const^save_12/MergeV2Checkpoints^save_12/control_dependency*
_output_shapes
: *
T0
ö
save_12/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
Ă
"save_12/RestoreV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
ű
save_12/RestoreV2	RestoreV2save_12/Constsave_12/RestoreV2/tensor_names"save_12/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
¤
save_12/AssignAssignbeta1_powersave_12/RestoreV2*
use_locking(*
T0*
_output_shapes
: *
validate_shape(* 
_class
loc:@vc/dense/bias
¨
save_12/Assign_1Assignbeta2_powersave_12/RestoreV2:1*
_output_shapes
: *
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
use_locking(
Ż
save_12/Assign_2Assignpi/dense/biassave_12/RestoreV2:2*
use_locking(*
T0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
validate_shape(
ˇ
save_12/Assign_3Assignpi/dense/kernelsave_12/RestoreV2:3*
T0*
validate_shape(*
_output_shapes
:	<*
use_locking(*"
_class
loc:@pi/dense/kernel
ł
save_12/Assign_4Assignpi/dense_1/biassave_12/RestoreV2:4*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:
ź
save_12/Assign_5Assignpi/dense_1/kernelsave_12/RestoreV2:5*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

˛
save_12/Assign_6Assignpi/dense_2/biassave_12/RestoreV2:6*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias
ť
save_12/Assign_7Assignpi/dense_2/kernelsave_12/RestoreV2:7*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
:	*
validate_shape(*
use_locking(
¨
save_12/Assign_8Assign
pi/log_stdsave_12/RestoreV2:8*
validate_shape(*
T0*
_output_shapes
:*
_class
loc:@pi/log_std*
use_locking(
Ż
save_12/Assign_9Assignvc/dense/biassave_12/RestoreV2:9*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(* 
_class
loc:@vc/dense/bias
ś
save_12/Assign_10Assignvc/dense/bias/Adamsave_12/RestoreV2:10*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:
¸
save_12/Assign_11Assignvc/dense/bias/Adam_1save_12/RestoreV2:11* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:
š
save_12/Assign_12Assignvc/dense/kernelsave_12/RestoreV2:12*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(*
T0
ž
save_12/Assign_13Assignvc/dense/kernel/Adamsave_12/RestoreV2:13*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
use_locking(*
validate_shape(*
T0
Ŕ
save_12/Assign_14Assignvc/dense/kernel/Adam_1save_12/RestoreV2:14*
validate_shape(*
use_locking(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
T0
ľ
save_12/Assign_15Assignvc/dense_1/biassave_12/RestoreV2:15*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(
ş
save_12/Assign_16Assignvc/dense_1/bias/Adamsave_12/RestoreV2:16*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(
ź
save_12/Assign_17Assignvc/dense_1/bias/Adam_1save_12/RestoreV2:17*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0
ž
save_12/Assign_18Assignvc/dense_1/kernelsave_12/RestoreV2:18* 
_output_shapes
:
*
T0*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
Ă
save_12/Assign_19Assignvc/dense_1/kernel/Adamsave_12/RestoreV2:19* 
_output_shapes
:
*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
use_locking(
Ĺ
save_12/Assign_20Assignvc/dense_1/kernel/Adam_1save_12/RestoreV2:20*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0
´
save_12/Assign_21Assignvc/dense_2/biassave_12/RestoreV2:21*
_output_shapes
:*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
T0
š
save_12/Assign_22Assignvc/dense_2/bias/Adamsave_12/RestoreV2:22*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0*
validate_shape(
ť
save_12/Assign_23Assignvc/dense_2/bias/Adam_1save_12/RestoreV2:23*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
˝
save_12/Assign_24Assignvc/dense_2/kernelsave_12/RestoreV2:24*
validate_shape(*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0
Â
save_12/Assign_25Assignvc/dense_2/kernel/Adamsave_12/RestoreV2:25*
_output_shapes
:	*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
T0
Ä
save_12/Assign_26Assignvc/dense_2/kernel/Adam_1save_12/RestoreV2:26*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
ą
save_12/Assign_27Assignvf/dense/biassave_12/RestoreV2:27*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@vf/dense/bias*
T0
ś
save_12/Assign_28Assignvf/dense/bias/Adamsave_12/RestoreV2:28* 
_class
loc:@vf/dense/bias*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(
¸
save_12/Assign_29Assignvf/dense/bias/Adam_1save_12/RestoreV2:29*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias
š
save_12/Assign_30Assignvf/dense/kernelsave_12/RestoreV2:30*
T0*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(
ž
save_12/Assign_31Assignvf/dense/kernel/Adamsave_12/RestoreV2:31*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
use_locking(
Ŕ
save_12/Assign_32Assignvf/dense/kernel/Adam_1save_12/RestoreV2:32*
validate_shape(*"
_class
loc:@vf/dense/kernel*
use_locking(*
T0*
_output_shapes
:	<
ľ
save_12/Assign_33Assignvf/dense_1/biassave_12/RestoreV2:33*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
ş
save_12/Assign_34Assignvf/dense_1/bias/Adamsave_12/RestoreV2:34*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0*
validate_shape(*
use_locking(
ź
save_12/Assign_35Assignvf/dense_1/bias/Adam_1save_12/RestoreV2:35*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(
ž
save_12/Assign_36Assignvf/dense_1/kernelsave_12/RestoreV2:36*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel
Ă
save_12/Assign_37Assignvf/dense_1/kernel/Adamsave_12/RestoreV2:37*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
Ĺ
save_12/Assign_38Assignvf/dense_1/kernel/Adam_1save_12/RestoreV2:38*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:

´
save_12/Assign_39Assignvf/dense_2/biassave_12/RestoreV2:39*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(
š
save_12/Assign_40Assignvf/dense_2/bias/Adamsave_12/RestoreV2:40*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_12/Assign_41Assignvf/dense_2/bias/Adam_1save_12/RestoreV2:41*
_output_shapes
:*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
T0*
use_locking(
˝
save_12/Assign_42Assignvf/dense_2/kernelsave_12/RestoreV2:42*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
Â
save_12/Assign_43Assignvf/dense_2/kernel/Adamsave_12/RestoreV2:43*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
Ä
save_12/Assign_44Assignvf/dense_2/kernel/Adam_1save_12/RestoreV2:44*
_output_shapes
:	*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(

save_12/restore_shardNoOp^save_12/Assign^save_12/Assign_1^save_12/Assign_10^save_12/Assign_11^save_12/Assign_12^save_12/Assign_13^save_12/Assign_14^save_12/Assign_15^save_12/Assign_16^save_12/Assign_17^save_12/Assign_18^save_12/Assign_19^save_12/Assign_2^save_12/Assign_20^save_12/Assign_21^save_12/Assign_22^save_12/Assign_23^save_12/Assign_24^save_12/Assign_25^save_12/Assign_26^save_12/Assign_27^save_12/Assign_28^save_12/Assign_29^save_12/Assign_3^save_12/Assign_30^save_12/Assign_31^save_12/Assign_32^save_12/Assign_33^save_12/Assign_34^save_12/Assign_35^save_12/Assign_36^save_12/Assign_37^save_12/Assign_38^save_12/Assign_39^save_12/Assign_4^save_12/Assign_40^save_12/Assign_41^save_12/Assign_42^save_12/Assign_43^save_12/Assign_44^save_12/Assign_5^save_12/Assign_6^save_12/Assign_7^save_12/Assign_8^save_12/Assign_9
3
save_12/restore_allNoOp^save_12/restore_shard
\
save_13/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_13/filenamePlaceholderWithDefaultsave_13/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_13/ConstPlaceholderWithDefaultsave_13/filename*
shape: *
dtype0*
_output_shapes
: 

save_13/StringJoin/inputs_1Const*<
value3B1 B+_temp_4fd8fe324daa44dba8ccf33741b0b7df/part*
dtype0*
_output_shapes
: 
~
save_13/StringJoin
StringJoinsave_13/Constsave_13/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_13/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_13/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0

save_13/ShardedFilenameShardedFilenamesave_13/StringJoinsave_13/ShardedFilename/shardsave_13/num_shards*
_output_shapes
: 
ó
save_13/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ŕ
save_13/SaveV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
ş
save_13/SaveV2SaveV2save_13/ShardedFilenamesave_13/SaveV2/tensor_namessave_13/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_13/control_dependencyIdentitysave_13/ShardedFilename^save_13/SaveV2**
_class 
loc:@save_13/ShardedFilename*
T0*
_output_shapes
: 
Ś
.save_13/MergeV2Checkpoints/checkpoint_prefixesPacksave_13/ShardedFilename^save_13/control_dependency*
N*
_output_shapes
:*

axis *
T0

save_13/MergeV2CheckpointsMergeV2Checkpoints.save_13/MergeV2Checkpoints/checkpoint_prefixessave_13/Const*
delete_old_dirs(

save_13/IdentityIdentitysave_13/Const^save_13/MergeV2Checkpoints^save_13/control_dependency*
T0*
_output_shapes
: 
ö
save_13/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
Ă
"save_13/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ű
save_13/RestoreV2	RestoreV2save_13/Constsave_13/RestoreV2/tensor_names"save_13/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
¤
save_13/AssignAssignbeta1_powersave_13/RestoreV2*
_output_shapes
: *
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(
¨
save_13/Assign_1Assignbeta2_powersave_13/RestoreV2:1*
validate_shape(*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias*
use_locking(
Ż
save_13/Assign_2Assignpi/dense/biassave_13/RestoreV2:2*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(* 
_class
loc:@pi/dense/bias
ˇ
save_13/Assign_3Assignpi/dense/kernelsave_13/RestoreV2:3*
T0*
_output_shapes
:	<*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel
ł
save_13/Assign_4Assignpi/dense_1/biassave_13/RestoreV2:4*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:
ź
save_13/Assign_5Assignpi/dense_1/kernelsave_13/RestoreV2:5*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
validate_shape(*
T0
˛
save_13/Assign_6Assignpi/dense_2/biassave_13/RestoreV2:6*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0*
validate_shape(
ť
save_13/Assign_7Assignpi/dense_2/kernelsave_13/RestoreV2:7*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	*
use_locking(
¨
save_13/Assign_8Assign
pi/log_stdsave_13/RestoreV2:8*
validate_shape(*
_output_shapes
:*
_class
loc:@pi/log_std*
use_locking(*
T0
Ż
save_13/Assign_9Assignvc/dense/biassave_13/RestoreV2:9*
T0* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:
ś
save_13/Assign_10Assignvc/dense/bias/Adamsave_13/RestoreV2:10*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
use_locking(
¸
save_13/Assign_11Assignvc/dense/bias/Adam_1save_13/RestoreV2:11*
T0*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:
š
save_13/Assign_12Assignvc/dense/kernelsave_13/RestoreV2:12*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel
ž
save_13/Assign_13Assignvc/dense/kernel/Adamsave_13/RestoreV2:13*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
use_locking(*
validate_shape(*
T0
Ŕ
save_13/Assign_14Assignvc/dense/kernel/Adam_1save_13/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	<*
use_locking(
ľ
save_13/Assign_15Assignvc/dense_1/biassave_13/RestoreV2:15*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
ş
save_13/Assign_16Assignvc/dense_1/bias/Adamsave_13/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_13/Assign_17Assignvc/dense_1/bias/Adam_1save_13/RestoreV2:17*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
ž
save_13/Assign_18Assignvc/dense_1/kernelsave_13/RestoreV2:18*
validate_shape(*
T0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
use_locking(
Ă
save_13/Assign_19Assignvc/dense_1/kernel/Adamsave_13/RestoreV2:19*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:
*
validate_shape(*
use_locking(
Ĺ
save_13/Assign_20Assignvc/dense_1/kernel/Adam_1save_13/RestoreV2:20*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
use_locking(
´
save_13/Assign_21Assignvc/dense_2/biassave_13/RestoreV2:21*
use_locking(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0
š
save_13/Assign_22Assignvc/dense_2/bias/Adamsave_13/RestoreV2:22*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(
ť
save_13/Assign_23Assignvc/dense_2/bias/Adam_1save_13/RestoreV2:23*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias
˝
save_13/Assign_24Assignvc/dense_2/kernelsave_13/RestoreV2:24*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
T0*
use_locking(
Â
save_13/Assign_25Assignvc/dense_2/kernel/Adamsave_13/RestoreV2:25*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0*
validate_shape(
Ä
save_13/Assign_26Assignvc/dense_2/kernel/Adam_1save_13/RestoreV2:26*
validate_shape(*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel
ą
save_13/Assign_27Assignvf/dense/biassave_13/RestoreV2:27*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(*
T0*
_output_shapes	
:
ś
save_13/Assign_28Assignvf/dense/bias/Adamsave_13/RestoreV2:28* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0
¸
save_13/Assign_29Assignvf/dense/bias/Adam_1save_13/RestoreV2:29* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
š
save_13/Assign_30Assignvf/dense/kernelsave_13/RestoreV2:30*
use_locking(*
_output_shapes
:	<*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel
ž
save_13/Assign_31Assignvf/dense/kernel/Adamsave_13/RestoreV2:31*
use_locking(*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@vf/dense/kernel*
T0
Ŕ
save_13/Assign_32Assignvf/dense/kernel/Adam_1save_13/RestoreV2:32*
_output_shapes
:	<*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(
ľ
save_13/Assign_33Assignvf/dense_1/biassave_13/RestoreV2:33*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0
ş
save_13/Assign_34Assignvf/dense_1/bias/Adamsave_13/RestoreV2:34*
T0*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(
ź
save_13/Assign_35Assignvf/dense_1/bias/Adam_1save_13/RestoreV2:35*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
ž
save_13/Assign_36Assignvf/dense_1/kernelsave_13/RestoreV2:36*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(
Ă
save_13/Assign_37Assignvf/dense_1/kernel/Adamsave_13/RestoreV2:37*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(*
use_locking(* 
_output_shapes
:

Ĺ
save_13/Assign_38Assignvf/dense_1/kernel/Adam_1save_13/RestoreV2:38*
T0*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:

´
save_13/Assign_39Assignvf/dense_2/biassave_13/RestoreV2:39*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias
š
save_13/Assign_40Assignvf/dense_2/bias/Adamsave_13/RestoreV2:40*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(
ť
save_13/Assign_41Assignvf/dense_2/bias/Adam_1save_13/RestoreV2:41*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(*
T0
˝
save_13/Assign_42Assignvf/dense_2/kernelsave_13/RestoreV2:42*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(
Â
save_13/Assign_43Assignvf/dense_2/kernel/Adamsave_13/RestoreV2:43*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0
Ä
save_13/Assign_44Assignvf/dense_2/kernel/Adam_1save_13/RestoreV2:44*
_output_shapes
:	*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel

save_13/restore_shardNoOp^save_13/Assign^save_13/Assign_1^save_13/Assign_10^save_13/Assign_11^save_13/Assign_12^save_13/Assign_13^save_13/Assign_14^save_13/Assign_15^save_13/Assign_16^save_13/Assign_17^save_13/Assign_18^save_13/Assign_19^save_13/Assign_2^save_13/Assign_20^save_13/Assign_21^save_13/Assign_22^save_13/Assign_23^save_13/Assign_24^save_13/Assign_25^save_13/Assign_26^save_13/Assign_27^save_13/Assign_28^save_13/Assign_29^save_13/Assign_3^save_13/Assign_30^save_13/Assign_31^save_13/Assign_32^save_13/Assign_33^save_13/Assign_34^save_13/Assign_35^save_13/Assign_36^save_13/Assign_37^save_13/Assign_38^save_13/Assign_39^save_13/Assign_4^save_13/Assign_40^save_13/Assign_41^save_13/Assign_42^save_13/Assign_43^save_13/Assign_44^save_13/Assign_5^save_13/Assign_6^save_13/Assign_7^save_13/Assign_8^save_13/Assign_9
3
save_13/restore_allNoOp^save_13/restore_shard
\
save_14/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_14/filenamePlaceholderWithDefaultsave_14/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_14/ConstPlaceholderWithDefaultsave_14/filename*
shape: *
dtype0*
_output_shapes
: 

save_14/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_efb0683405f34c198ad6ae81b85d8fd6/part*
dtype0
~
save_14/StringJoin
StringJoinsave_14/Constsave_14/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_14/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_14/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_14/ShardedFilenameShardedFilenamesave_14/StringJoinsave_14/ShardedFilename/shardsave_14/num_shards*
_output_shapes
: 
ó
save_14/SaveV2/tensor_namesConst*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-
Ŕ
save_14/SaveV2/shape_and_slicesConst*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-
ş
save_14/SaveV2SaveV2save_14/ShardedFilenamesave_14/SaveV2/tensor_namessave_14/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_14/control_dependencyIdentitysave_14/ShardedFilename^save_14/SaveV2*
_output_shapes
: **
_class 
loc:@save_14/ShardedFilename*
T0
Ś
.save_14/MergeV2Checkpoints/checkpoint_prefixesPacksave_14/ShardedFilename^save_14/control_dependency*

axis *
_output_shapes
:*
N*
T0

save_14/MergeV2CheckpointsMergeV2Checkpoints.save_14/MergeV2Checkpoints/checkpoint_prefixessave_14/Const*
delete_old_dirs(

save_14/IdentityIdentitysave_14/Const^save_14/MergeV2Checkpoints^save_14/control_dependency*
T0*
_output_shapes
: 
ö
save_14/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
Ă
"save_14/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ű
save_14/RestoreV2	RestoreV2save_14/Constsave_14/RestoreV2/tensor_names"save_14/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
¤
save_14/AssignAssignbeta1_powersave_14/RestoreV2*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: *
validate_shape(
¨
save_14/Assign_1Assignbeta2_powersave_14/RestoreV2:1*
_output_shapes
: *
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(
Ż
save_14/Assign_2Assignpi/dense/biassave_14/RestoreV2:2*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
_output_shapes	
:
ˇ
save_14/Assign_3Assignpi/dense/kernelsave_14/RestoreV2:3*
use_locking(*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0
ł
save_14/Assign_4Assignpi/dense_1/biassave_14/RestoreV2:4*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias
ź
save_14/Assign_5Assignpi/dense_1/kernelsave_14/RestoreV2:5*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0
˛
save_14/Assign_6Assignpi/dense_2/biassave_14/RestoreV2:6*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias
ť
save_14/Assign_7Assignpi/dense_2/kernelsave_14/RestoreV2:7*
validate_shape(*
_output_shapes
:	*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0
¨
save_14/Assign_8Assign
pi/log_stdsave_14/RestoreV2:8*
T0*
_class
loc:@pi/log_std*
use_locking(*
validate_shape(*
_output_shapes
:
Ż
save_14/Assign_9Assignvc/dense/biassave_14/RestoreV2:9*
_output_shapes	
:*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
T0
ś
save_14/Assign_10Assignvc/dense/bias/Adamsave_14/RestoreV2:10* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
¸
save_14/Assign_11Assignvc/dense/bias/Adam_1save_14/RestoreV2:11* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
š
save_14/Assign_12Assignvc/dense/kernelsave_14/RestoreV2:12*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	<*
use_locking(
ž
save_14/Assign_13Assignvc/dense/kernel/Adamsave_14/RestoreV2:13*
validate_shape(*
T0*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
use_locking(
Ŕ
save_14/Assign_14Assignvc/dense/kernel/Adam_1save_14/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
T0*
use_locking(*
_output_shapes
:	<*
validate_shape(
ľ
save_14/Assign_15Assignvc/dense_1/biassave_14/RestoreV2:15*
use_locking(*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0
ş
save_14/Assign_16Assignvc/dense_1/bias/Adamsave_14/RestoreV2:16*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:*
use_locking(
ź
save_14/Assign_17Assignvc/dense_1/bias/Adam_1save_14/RestoreV2:17*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
ž
save_14/Assign_18Assignvc/dense_1/kernelsave_14/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:
*
validate_shape(*
use_locking(
Ă
save_14/Assign_19Assignvc/dense_1/kernel/Adamsave_14/RestoreV2:19*
use_locking(*
T0* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
Ĺ
save_14/Assign_20Assignvc/dense_1/kernel/Adam_1save_14/RestoreV2:20*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0
´
save_14/Assign_21Assignvc/dense_2/biassave_14/RestoreV2:21*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
š
save_14/Assign_22Assignvc/dense_2/bias/Adamsave_14/RestoreV2:22*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(*
validate_shape(*
T0
ť
save_14/Assign_23Assignvc/dense_2/bias/Adam_1save_14/RestoreV2:23*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(
˝
save_14/Assign_24Assignvc/dense_2/kernelsave_14/RestoreV2:24*
_output_shapes
:	*
T0*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(
Â
save_14/Assign_25Assignvc/dense_2/kernel/Adamsave_14/RestoreV2:25*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	
Ä
save_14/Assign_26Assignvc/dense_2/kernel/Adam_1save_14/RestoreV2:26*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel
ą
save_14/Assign_27Assignvf/dense/biassave_14/RestoreV2:27*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(* 
_class
loc:@vf/dense/bias
ś
save_14/Assign_28Assignvf/dense/bias/Adamsave_14/RestoreV2:28*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
¸
save_14/Assign_29Assignvf/dense/bias/Adam_1save_14/RestoreV2:29*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias
š
save_14/Assign_30Assignvf/dense/kernelsave_14/RestoreV2:30*
T0*
_output_shapes
:	<*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense/kernel
ž
save_14/Assign_31Assignvf/dense/kernel/Adamsave_14/RestoreV2:31*
_output_shapes
:	<*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense/kernel
Ŕ
save_14/Assign_32Assignvf/dense/kernel/Adam_1save_14/RestoreV2:32*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
validate_shape(*
T0*
use_locking(
ľ
save_14/Assign_33Assignvf/dense_1/biassave_14/RestoreV2:33*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(
ş
save_14/Assign_34Assignvf/dense_1/bias/Adamsave_14/RestoreV2:34*
T0*
use_locking(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
validate_shape(
ź
save_14/Assign_35Assignvf/dense_1/bias/Adam_1save_14/RestoreV2:35*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0
ž
save_14/Assign_36Assignvf/dense_1/kernelsave_14/RestoreV2:36*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(
Ă
save_14/Assign_37Assignvf/dense_1/kernel/Adamsave_14/RestoreV2:37*
validate_shape(*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(
Ĺ
save_14/Assign_38Assignvf/dense_1/kernel/Adam_1save_14/RestoreV2:38* 
_output_shapes
:
*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel
´
save_14/Assign_39Assignvf/dense_2/biassave_14/RestoreV2:39*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias
š
save_14/Assign_40Assignvf/dense_2/bias/Adamsave_14/RestoreV2:40*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
ť
save_14/Assign_41Assignvf/dense_2/bias/Adam_1save_14/RestoreV2:41*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
˝
save_14/Assign_42Assignvf/dense_2/kernelsave_14/RestoreV2:42*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(
Â
save_14/Assign_43Assignvf/dense_2/kernel/Adamsave_14/RestoreV2:43*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0
Ä
save_14/Assign_44Assignvf/dense_2/kernel/Adam_1save_14/RestoreV2:44*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	

save_14/restore_shardNoOp^save_14/Assign^save_14/Assign_1^save_14/Assign_10^save_14/Assign_11^save_14/Assign_12^save_14/Assign_13^save_14/Assign_14^save_14/Assign_15^save_14/Assign_16^save_14/Assign_17^save_14/Assign_18^save_14/Assign_19^save_14/Assign_2^save_14/Assign_20^save_14/Assign_21^save_14/Assign_22^save_14/Assign_23^save_14/Assign_24^save_14/Assign_25^save_14/Assign_26^save_14/Assign_27^save_14/Assign_28^save_14/Assign_29^save_14/Assign_3^save_14/Assign_30^save_14/Assign_31^save_14/Assign_32^save_14/Assign_33^save_14/Assign_34^save_14/Assign_35^save_14/Assign_36^save_14/Assign_37^save_14/Assign_38^save_14/Assign_39^save_14/Assign_4^save_14/Assign_40^save_14/Assign_41^save_14/Assign_42^save_14/Assign_43^save_14/Assign_44^save_14/Assign_5^save_14/Assign_6^save_14/Assign_7^save_14/Assign_8^save_14/Assign_9
3
save_14/restore_allNoOp^save_14/restore_shard
\
save_15/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_15/filenamePlaceholderWithDefaultsave_15/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_15/ConstPlaceholderWithDefaultsave_15/filename*
_output_shapes
: *
dtype0*
shape: 

save_15/StringJoin/inputs_1Const*<
value3B1 B+_temp_2d357bc6343f4e1daca6e59726683a21/part*
_output_shapes
: *
dtype0
~
save_15/StringJoin
StringJoinsave_15/Constsave_15/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_15/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_15/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_15/ShardedFilenameShardedFilenamesave_15/StringJoinsave_15/ShardedFilename/shardsave_15/num_shards*
_output_shapes
: 
ó
save_15/SaveV2/tensor_namesConst*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
Ŕ
save_15/SaveV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ş
save_15/SaveV2SaveV2save_15/ShardedFilenamesave_15/SaveV2/tensor_namessave_15/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_15/control_dependencyIdentitysave_15/ShardedFilename^save_15/SaveV2**
_class 
loc:@save_15/ShardedFilename*
_output_shapes
: *
T0
Ś
.save_15/MergeV2Checkpoints/checkpoint_prefixesPacksave_15/ShardedFilename^save_15/control_dependency*

axis *
_output_shapes
:*
N*
T0

save_15/MergeV2CheckpointsMergeV2Checkpoints.save_15/MergeV2Checkpoints/checkpoint_prefixessave_15/Const*
delete_old_dirs(

save_15/IdentityIdentitysave_15/Const^save_15/MergeV2Checkpoints^save_15/control_dependency*
_output_shapes
: *
T0
ö
save_15/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ă
"save_15/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ű
save_15/RestoreV2	RestoreV2save_15/Constsave_15/RestoreV2/tensor_names"save_15/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
¤
save_15/AssignAssignbeta1_powersave_15/RestoreV2*
_output_shapes
: *
validate_shape(*
use_locking(*
T0* 
_class
loc:@vc/dense/bias
¨
save_15/Assign_1Assignbeta2_powersave_15/RestoreV2:1*
_output_shapes
: *
use_locking(*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias
Ż
save_15/Assign_2Assignpi/dense/biassave_15/RestoreV2:2*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@pi/dense/bias
ˇ
save_15/Assign_3Assignpi/dense/kernelsave_15/RestoreV2:3*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes
:	<
ł
save_15/Assign_4Assignpi/dense_1/biassave_15/RestoreV2:4*
T0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(
ź
save_15/Assign_5Assignpi/dense_1/kernelsave_15/RestoreV2:5*
validate_shape(*
T0* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
use_locking(
˛
save_15/Assign_6Assignpi/dense_2/biassave_15/RestoreV2:6*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(*
validate_shape(
ť
save_15/Assign_7Assignpi/dense_2/kernelsave_15/RestoreV2:7*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	
¨
save_15/Assign_8Assign
pi/log_stdsave_15/RestoreV2:8*
use_locking(*
_output_shapes
:*
_class
loc:@pi/log_std*
validate_shape(*
T0
Ż
save_15/Assign_9Assignvc/dense/biassave_15/RestoreV2:9*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@vc/dense/bias*
T0
ś
save_15/Assign_10Assignvc/dense/bias/Adamsave_15/RestoreV2:10*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
use_locking(
¸
save_15/Assign_11Assignvc/dense/bias/Adam_1save_15/RestoreV2:11*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vc/dense/bias
š
save_15/Assign_12Assignvc/dense/kernelsave_15/RestoreV2:12*
validate_shape(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
T0*
use_locking(
ž
save_15/Assign_13Assignvc/dense/kernel/Adamsave_15/RestoreV2:13*
T0*
_output_shapes
:	<*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel
Ŕ
save_15/Assign_14Assignvc/dense/kernel/Adam_1save_15/RestoreV2:14*
_output_shapes
:	<*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense/kernel
ľ
save_15/Assign_15Assignvc/dense_1/biassave_15/RestoreV2:15*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
ş
save_15/Assign_16Assignvc/dense_1/bias/Adamsave_15/RestoreV2:16*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
ź
save_15/Assign_17Assignvc/dense_1/bias/Adam_1save_15/RestoreV2:17*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias
ž
save_15/Assign_18Assignvc/dense_1/kernelsave_15/RestoreV2:18* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(*
T0
Ă
save_15/Assign_19Assignvc/dense_1/kernel/Adamsave_15/RestoreV2:19*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(
Ĺ
save_15/Assign_20Assignvc/dense_1/kernel/Adam_1save_15/RestoreV2:20*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
´
save_15/Assign_21Assignvc/dense_2/biassave_15/RestoreV2:21*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
š
save_15/Assign_22Assignvc/dense_2/bias/Adamsave_15/RestoreV2:22*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
ť
save_15/Assign_23Assignvc/dense_2/bias/Adam_1save_15/RestoreV2:23*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias
˝
save_15/Assign_24Assignvc/dense_2/kernelsave_15/RestoreV2:24*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
Â
save_15/Assign_25Assignvc/dense_2/kernel/Adamsave_15/RestoreV2:25*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	
Ä
save_15/Assign_26Assignvc/dense_2/kernel/Adam_1save_15/RestoreV2:26*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	
ą
save_15/Assign_27Assignvf/dense/biassave_15/RestoreV2:27*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
ś
save_15/Assign_28Assignvf/dense/bias/Adamsave_15/RestoreV2:28*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
¸
save_15/Assign_29Assignvf/dense/bias/Adam_1save_15/RestoreV2:29* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(
š
save_15/Assign_30Assignvf/dense/kernelsave_15/RestoreV2:30*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel
ž
save_15/Assign_31Assignvf/dense/kernel/Adamsave_15/RestoreV2:31*
_output_shapes
:	<*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel
Ŕ
save_15/Assign_32Assignvf/dense/kernel/Adam_1save_15/RestoreV2:32*
use_locking(*
T0*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
validate_shape(
ľ
save_15/Assign_33Assignvf/dense_1/biassave_15/RestoreV2:33*
T0*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
use_locking(
ş
save_15/Assign_34Assignvf/dense_1/bias/Adamsave_15/RestoreV2:34*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
ź
save_15/Assign_35Assignvf/dense_1/bias/Adam_1save_15/RestoreV2:35*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0
ž
save_15/Assign_36Assignvf/dense_1/kernelsave_15/RestoreV2:36* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0
Ă
save_15/Assign_37Assignvf/dense_1/kernel/Adamsave_15/RestoreV2:37* 
_output_shapes
:
*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel
Ĺ
save_15/Assign_38Assignvf/dense_1/kernel/Adam_1save_15/RestoreV2:38*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
´
save_15/Assign_39Assignvf/dense_2/biassave_15/RestoreV2:39*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0*
use_locking(
š
save_15/Assign_40Assignvf/dense_2/bias/Adamsave_15/RestoreV2:40*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
ť
save_15/Assign_41Assignvf/dense_2/bias/Adam_1save_15/RestoreV2:41*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(
˝
save_15/Assign_42Assignvf/dense_2/kernelsave_15/RestoreV2:42*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel
Â
save_15/Assign_43Assignvf/dense_2/kernel/Adamsave_15/RestoreV2:43*
T0*
use_locking(*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
Ä
save_15/Assign_44Assignvf/dense_2/kernel/Adam_1save_15/RestoreV2:44*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0*
_output_shapes
:	*
use_locking(

save_15/restore_shardNoOp^save_15/Assign^save_15/Assign_1^save_15/Assign_10^save_15/Assign_11^save_15/Assign_12^save_15/Assign_13^save_15/Assign_14^save_15/Assign_15^save_15/Assign_16^save_15/Assign_17^save_15/Assign_18^save_15/Assign_19^save_15/Assign_2^save_15/Assign_20^save_15/Assign_21^save_15/Assign_22^save_15/Assign_23^save_15/Assign_24^save_15/Assign_25^save_15/Assign_26^save_15/Assign_27^save_15/Assign_28^save_15/Assign_29^save_15/Assign_3^save_15/Assign_30^save_15/Assign_31^save_15/Assign_32^save_15/Assign_33^save_15/Assign_34^save_15/Assign_35^save_15/Assign_36^save_15/Assign_37^save_15/Assign_38^save_15/Assign_39^save_15/Assign_4^save_15/Assign_40^save_15/Assign_41^save_15/Assign_42^save_15/Assign_43^save_15/Assign_44^save_15/Assign_5^save_15/Assign_6^save_15/Assign_7^save_15/Assign_8^save_15/Assign_9
3
save_15/restore_allNoOp^save_15/restore_shard
\
save_16/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_16/filenamePlaceholderWithDefaultsave_16/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_16/ConstPlaceholderWithDefaultsave_16/filename*
_output_shapes
: *
dtype0*
shape: 

save_16/StringJoin/inputs_1Const*<
value3B1 B+_temp_276a408352734c949e3ca92027df27d9/part*
_output_shapes
: *
dtype0
~
save_16/StringJoin
StringJoinsave_16/Constsave_16/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_16/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_16/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_16/ShardedFilenameShardedFilenamesave_16/StringJoinsave_16/ShardedFilename/shardsave_16/num_shards*
_output_shapes
: 
ó
save_16/SaveV2/tensor_namesConst*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
Ŕ
save_16/SaveV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ş
save_16/SaveV2SaveV2save_16/ShardedFilenamesave_16/SaveV2/tensor_namessave_16/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_16/control_dependencyIdentitysave_16/ShardedFilename^save_16/SaveV2**
_class 
loc:@save_16/ShardedFilename*
T0*
_output_shapes
: 
Ś
.save_16/MergeV2Checkpoints/checkpoint_prefixesPacksave_16/ShardedFilename^save_16/control_dependency*
N*
_output_shapes
:*
T0*

axis 

save_16/MergeV2CheckpointsMergeV2Checkpoints.save_16/MergeV2Checkpoints/checkpoint_prefixessave_16/Const*
delete_old_dirs(

save_16/IdentityIdentitysave_16/Const^save_16/MergeV2Checkpoints^save_16/control_dependency*
_output_shapes
: *
T0
ö
save_16/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
Ă
"save_16/RestoreV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
ű
save_16/RestoreV2	RestoreV2save_16/Constsave_16/RestoreV2/tensor_names"save_16/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
¤
save_16/AssignAssignbeta1_powersave_16/RestoreV2*
T0*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes
: 
¨
save_16/Assign_1Assignbeta2_powersave_16/RestoreV2:1*
_output_shapes
: *
use_locking(*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias
Ż
save_16/Assign_2Assignpi/dense/biassave_16/RestoreV2:2*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0*
use_locking(
ˇ
save_16/Assign_3Assignpi/dense/kernelsave_16/RestoreV2:3*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	<
ł
save_16/Assign_4Assignpi/dense_1/biassave_16/RestoreV2:4*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:
ź
save_16/Assign_5Assignpi/dense_1/kernelsave_16/RestoreV2:5*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(* 
_output_shapes
:

˛
save_16/Assign_6Assignpi/dense_2/biassave_16/RestoreV2:6*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_16/Assign_7Assignpi/dense_2/kernelsave_16/RestoreV2:7*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(
¨
save_16/Assign_8Assign
pi/log_stdsave_16/RestoreV2:8*
_class
loc:@pi/log_std*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
Ż
save_16/Assign_9Assignvc/dense/biassave_16/RestoreV2:9* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
ś
save_16/Assign_10Assignvc/dense/bias/Adamsave_16/RestoreV2:10* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
¸
save_16/Assign_11Assignvc/dense/bias/Adam_1save_16/RestoreV2:11*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes	
:
š
save_16/Assign_12Assignvc/dense/kernelsave_16/RestoreV2:12*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
validate_shape(*
T0
ž
save_16/Assign_13Assignvc/dense/kernel/Adamsave_16/RestoreV2:13*
_output_shapes
:	<*
T0*
use_locking(*"
_class
loc:@vc/dense/kernel*
validate_shape(
Ŕ
save_16/Assign_14Assignvc/dense/kernel/Adam_1save_16/RestoreV2:14*
use_locking(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0
ľ
save_16/Assign_15Assignvc/dense_1/biassave_16/RestoreV2:15*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_16/Assign_16Assignvc/dense_1/bias/Adamsave_16/RestoreV2:16*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
use_locking(*
T0*
validate_shape(
ź
save_16/Assign_17Assignvc/dense_1/bias/Adam_1save_16/RestoreV2:17*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ž
save_16/Assign_18Assignvc/dense_1/kernelsave_16/RestoreV2:18* 
_output_shapes
:
*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(
Ă
save_16/Assign_19Assignvc/dense_1/kernel/Adamsave_16/RestoreV2:19*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:

Ĺ
save_16/Assign_20Assignvc/dense_1/kernel/Adam_1save_16/RestoreV2:20* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
validate_shape(
´
save_16/Assign_21Assignvc/dense_2/biassave_16/RestoreV2:21*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0
š
save_16/Assign_22Assignvc/dense_2/bias/Adamsave_16/RestoreV2:22*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
use_locking(
ť
save_16/Assign_23Assignvc/dense_2/bias/Adam_1save_16/RestoreV2:23*
use_locking(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
˝
save_16/Assign_24Assignvc/dense_2/kernelsave_16/RestoreV2:24*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
T0*
validate_shape(
Â
save_16/Assign_25Assignvc/dense_2/kernel/Adamsave_16/RestoreV2:25*
_output_shapes
:	*
T0*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(
Ä
save_16/Assign_26Assignvc/dense_2/kernel/Adam_1save_16/RestoreV2:26*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(
ą
save_16/Assign_27Assignvf/dense/biassave_16/RestoreV2:27* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0
ś
save_16/Assign_28Assignvf/dense/bias/Adamsave_16/RestoreV2:28* 
_class
loc:@vf/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:
¸
save_16/Assign_29Assignvf/dense/bias/Adam_1save_16/RestoreV2:29* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0
š
save_16/Assign_30Assignvf/dense/kernelsave_16/RestoreV2:30*
T0*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
use_locking(
ž
save_16/Assign_31Assignvf/dense/kernel/Adamsave_16/RestoreV2:31*
_output_shapes
:	<*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel
Ŕ
save_16/Assign_32Assignvf/dense/kernel/Adam_1save_16/RestoreV2:32*"
_class
loc:@vf/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	<*
use_locking(
ľ
save_16/Assign_33Assignvf/dense_1/biassave_16/RestoreV2:33*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0
ş
save_16/Assign_34Assignvf/dense_1/bias/Adamsave_16/RestoreV2:34*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias
ź
save_16/Assign_35Assignvf/dense_1/bias/Adam_1save_16/RestoreV2:35*
T0*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
use_locking(
ž
save_16/Assign_36Assignvf/dense_1/kernelsave_16/RestoreV2:36*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
use_locking(
Ă
save_16/Assign_37Assignvf/dense_1/kernel/Adamsave_16/RestoreV2:37*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
Ĺ
save_16/Assign_38Assignvf/dense_1/kernel/Adam_1save_16/RestoreV2:38* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(
´
save_16/Assign_39Assignvf/dense_2/biassave_16/RestoreV2:39*"
_class
loc:@vf/dense_2/bias*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
š
save_16/Assign_40Assignvf/dense_2/bias/Adamsave_16/RestoreV2:40*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0
ť
save_16/Assign_41Assignvf/dense_2/bias/Adam_1save_16/RestoreV2:41*
use_locking(*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0
˝
save_16/Assign_42Assignvf/dense_2/kernelsave_16/RestoreV2:42*
_output_shapes
:	*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
use_locking(
Â
save_16/Assign_43Assignvf/dense_2/kernel/Adamsave_16/RestoreV2:43*
_output_shapes
:	*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0*
validate_shape(
Ä
save_16/Assign_44Assignvf/dense_2/kernel/Adam_1save_16/RestoreV2:44*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0

save_16/restore_shardNoOp^save_16/Assign^save_16/Assign_1^save_16/Assign_10^save_16/Assign_11^save_16/Assign_12^save_16/Assign_13^save_16/Assign_14^save_16/Assign_15^save_16/Assign_16^save_16/Assign_17^save_16/Assign_18^save_16/Assign_19^save_16/Assign_2^save_16/Assign_20^save_16/Assign_21^save_16/Assign_22^save_16/Assign_23^save_16/Assign_24^save_16/Assign_25^save_16/Assign_26^save_16/Assign_27^save_16/Assign_28^save_16/Assign_29^save_16/Assign_3^save_16/Assign_30^save_16/Assign_31^save_16/Assign_32^save_16/Assign_33^save_16/Assign_34^save_16/Assign_35^save_16/Assign_36^save_16/Assign_37^save_16/Assign_38^save_16/Assign_39^save_16/Assign_4^save_16/Assign_40^save_16/Assign_41^save_16/Assign_42^save_16/Assign_43^save_16/Assign_44^save_16/Assign_5^save_16/Assign_6^save_16/Assign_7^save_16/Assign_8^save_16/Assign_9
3
save_16/restore_allNoOp^save_16/restore_shard
\
save_17/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_17/filenamePlaceholderWithDefaultsave_17/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_17/ConstPlaceholderWithDefaultsave_17/filename*
dtype0*
_output_shapes
: *
shape: 

save_17/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_085b68080747459bb1c269d76c29afbe/part
~
save_17/StringJoin
StringJoinsave_17/Constsave_17/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_17/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_17/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_17/ShardedFilenameShardedFilenamesave_17/StringJoinsave_17/ShardedFilename/shardsave_17/num_shards*
_output_shapes
: 
ó
save_17/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ŕ
save_17/SaveV2/shape_and_slicesConst*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-
ş
save_17/SaveV2SaveV2save_17/ShardedFilenamesave_17/SaveV2/tensor_namessave_17/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_17/control_dependencyIdentitysave_17/ShardedFilename^save_17/SaveV2*
_output_shapes
: **
_class 
loc:@save_17/ShardedFilename*
T0
Ś
.save_17/MergeV2Checkpoints/checkpoint_prefixesPacksave_17/ShardedFilename^save_17/control_dependency*
T0*

axis *
_output_shapes
:*
N

save_17/MergeV2CheckpointsMergeV2Checkpoints.save_17/MergeV2Checkpoints/checkpoint_prefixessave_17/Const*
delete_old_dirs(

save_17/IdentityIdentitysave_17/Const^save_17/MergeV2Checkpoints^save_17/control_dependency*
_output_shapes
: *
T0
ö
save_17/RestoreV2/tensor_namesConst*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
Ă
"save_17/RestoreV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
ű
save_17/RestoreV2	RestoreV2save_17/Constsave_17/RestoreV2/tensor_names"save_17/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
¤
save_17/AssignAssignbeta1_powersave_17/RestoreV2* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
¨
save_17/Assign_1Assignbeta2_powersave_17/RestoreV2:1*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
use_locking(
Ż
save_17/Assign_2Assignpi/dense/biassave_17/RestoreV2:2*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
validate_shape(
ˇ
save_17/Assign_3Assignpi/dense/kernelsave_17/RestoreV2:3*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
T0*
use_locking(
ł
save_17/Assign_4Assignpi/dense_1/biassave_17/RestoreV2:4*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(
ź
save_17/Assign_5Assignpi/dense_1/kernelsave_17/RestoreV2:5*
validate_shape(* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(
˛
save_17/Assign_6Assignpi/dense_2/biassave_17/RestoreV2:6*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
ť
save_17/Assign_7Assignpi/dense_2/kernelsave_17/RestoreV2:7*
validate_shape(*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@pi/dense_2/kernel
¨
save_17/Assign_8Assign
pi/log_stdsave_17/RestoreV2:8*
T0*
_class
loc:@pi/log_std*
use_locking(*
_output_shapes
:*
validate_shape(
Ż
save_17/Assign_9Assignvc/dense/biassave_17/RestoreV2:9*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias
ś
save_17/Assign_10Assignvc/dense/bias/Adamsave_17/RestoreV2:10*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias
¸
save_17/Assign_11Assignvc/dense/bias/Adam_1save_17/RestoreV2:11*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0
š
save_17/Assign_12Assignvc/dense/kernelsave_17/RestoreV2:12*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
T0*
validate_shape(
ž
save_17/Assign_13Assignvc/dense/kernel/Adamsave_17/RestoreV2:13*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
validate_shape(
Ŕ
save_17/Assign_14Assignvc/dense/kernel/Adam_1save_17/RestoreV2:14*
_output_shapes
:	<*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(
ľ
save_17/Assign_15Assignvc/dense_1/biassave_17/RestoreV2:15*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(
ş
save_17/Assign_16Assignvc/dense_1/bias/Adamsave_17/RestoreV2:16*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
use_locking(
ź
save_17/Assign_17Assignvc/dense_1/bias/Adam_1save_17/RestoreV2:17*
use_locking(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
T0
ž
save_17/Assign_18Assignvc/dense_1/kernelsave_17/RestoreV2:18*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0
Ă
save_17/Assign_19Assignvc/dense_1/kernel/Adamsave_17/RestoreV2:19* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(*
use_locking(
Ĺ
save_17/Assign_20Assignvc/dense_1/kernel/Adam_1save_17/RestoreV2:20*
use_locking(* 
_output_shapes
:
*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel
´
save_17/Assign_21Assignvc/dense_2/biassave_17/RestoreV2:21*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0
š
save_17/Assign_22Assignvc/dense_2/bias/Adamsave_17/RestoreV2:22*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
ť
save_17/Assign_23Assignvc/dense_2/bias/Adam_1save_17/RestoreV2:23*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(
˝
save_17/Assign_24Assignvc/dense_2/kernelsave_17/RestoreV2:24*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(
Â
save_17/Assign_25Assignvc/dense_2/kernel/Adamsave_17/RestoreV2:25*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	
Ä
save_17/Assign_26Assignvc/dense_2/kernel/Adam_1save_17/RestoreV2:26*
validate_shape(*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(
ą
save_17/Assign_27Assignvf/dense/biassave_17/RestoreV2:27*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias
ś
save_17/Assign_28Assignvf/dense/bias/Adamsave_17/RestoreV2:28* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
¸
save_17/Assign_29Assignvf/dense/bias/Adam_1save_17/RestoreV2:29*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias
š
save_17/Assign_30Assignvf/dense/kernelsave_17/RestoreV2:30*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	<
ž
save_17/Assign_31Assignvf/dense/kernel/Adamsave_17/RestoreV2:31*
T0*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@vf/dense/kernel*
use_locking(
Ŕ
save_17/Assign_32Assignvf/dense/kernel/Adam_1save_17/RestoreV2:32*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
T0
ľ
save_17/Assign_33Assignvf/dense_1/biassave_17/RestoreV2:33*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(*
validate_shape(
ş
save_17/Assign_34Assignvf/dense_1/bias/Adamsave_17/RestoreV2:34*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias
ź
save_17/Assign_35Assignvf/dense_1/bias/Adam_1save_17/RestoreV2:35*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(
ž
save_17/Assign_36Assignvf/dense_1/kernelsave_17/RestoreV2:36*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:
*
validate_shape(*
use_locking(
Ă
save_17/Assign_37Assignvf/dense_1/kernel/Adamsave_17/RestoreV2:37*
use_locking(* 
_output_shapes
:
*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel
Ĺ
save_17/Assign_38Assignvf/dense_1/kernel/Adam_1save_17/RestoreV2:38*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
use_locking(
´
save_17/Assign_39Assignvf/dense_2/biassave_17/RestoreV2:39*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(
š
save_17/Assign_40Assignvf/dense_2/bias/Adamsave_17/RestoreV2:40*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
ť
save_17/Assign_41Assignvf/dense_2/bias/Adam_1save_17/RestoreV2:41*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
˝
save_17/Assign_42Assignvf/dense_2/kernelsave_17/RestoreV2:42*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	
Â
save_17/Assign_43Assignvf/dense_2/kernel/Adamsave_17/RestoreV2:43*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
Ä
save_17/Assign_44Assignvf/dense_2/kernel/Adam_1save_17/RestoreV2:44*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0*
use_locking(

save_17/restore_shardNoOp^save_17/Assign^save_17/Assign_1^save_17/Assign_10^save_17/Assign_11^save_17/Assign_12^save_17/Assign_13^save_17/Assign_14^save_17/Assign_15^save_17/Assign_16^save_17/Assign_17^save_17/Assign_18^save_17/Assign_19^save_17/Assign_2^save_17/Assign_20^save_17/Assign_21^save_17/Assign_22^save_17/Assign_23^save_17/Assign_24^save_17/Assign_25^save_17/Assign_26^save_17/Assign_27^save_17/Assign_28^save_17/Assign_29^save_17/Assign_3^save_17/Assign_30^save_17/Assign_31^save_17/Assign_32^save_17/Assign_33^save_17/Assign_34^save_17/Assign_35^save_17/Assign_36^save_17/Assign_37^save_17/Assign_38^save_17/Assign_39^save_17/Assign_4^save_17/Assign_40^save_17/Assign_41^save_17/Assign_42^save_17/Assign_43^save_17/Assign_44^save_17/Assign_5^save_17/Assign_6^save_17/Assign_7^save_17/Assign_8^save_17/Assign_9
3
save_17/restore_allNoOp^save_17/restore_shard
\
save_18/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_18/filenamePlaceholderWithDefaultsave_18/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_18/ConstPlaceholderWithDefaultsave_18/filename*
shape: *
dtype0*
_output_shapes
: 

save_18/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_ff7b46fd78c34a28becec8c8f312c05b/part
~
save_18/StringJoin
StringJoinsave_18/Constsave_18/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
T
save_18/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_18/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 

save_18/ShardedFilenameShardedFilenamesave_18/StringJoinsave_18/ShardedFilename/shardsave_18/num_shards*
_output_shapes
: 
ó
save_18/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ŕ
save_18/SaveV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
ş
save_18/SaveV2SaveV2save_18/ShardedFilenamesave_18/SaveV2/tensor_namessave_18/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_18/control_dependencyIdentitysave_18/ShardedFilename^save_18/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_18/ShardedFilename
Ś
.save_18/MergeV2Checkpoints/checkpoint_prefixesPacksave_18/ShardedFilename^save_18/control_dependency*
N*
_output_shapes
:*
T0*

axis 

save_18/MergeV2CheckpointsMergeV2Checkpoints.save_18/MergeV2Checkpoints/checkpoint_prefixessave_18/Const*
delete_old_dirs(

save_18/IdentityIdentitysave_18/Const^save_18/MergeV2Checkpoints^save_18/control_dependency*
T0*
_output_shapes
: 
ö
save_18/RestoreV2/tensor_namesConst*
_output_shapes
:-*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ă
"save_18/RestoreV2/shape_and_slicesConst*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-
ű
save_18/RestoreV2	RestoreV2save_18/Constsave_18/RestoreV2/tensor_names"save_18/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
¤
save_18/AssignAssignbeta1_powersave_18/RestoreV2*
validate_shape(*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias*
use_locking(
¨
save_18/Assign_1Assignbeta2_powersave_18/RestoreV2:1*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes
: *
T0
Ż
save_18/Assign_2Assignpi/dense/biassave_18/RestoreV2:2*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
ˇ
save_18/Assign_3Assignpi/dense/kernelsave_18/RestoreV2:3*
validate_shape(*
T0*
_output_shapes
:	<*
use_locking(*"
_class
loc:@pi/dense/kernel
ł
save_18/Assign_4Assignpi/dense_1/biassave_18/RestoreV2:4*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@pi/dense_1/bias
ź
save_18/Assign_5Assignpi/dense_1/kernelsave_18/RestoreV2:5*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

˛
save_18/Assign_6Assignpi/dense_2/biassave_18/RestoreV2:6*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:
ť
save_18/Assign_7Assignpi/dense_2/kernelsave_18/RestoreV2:7*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
¨
save_18/Assign_8Assign
pi/log_stdsave_18/RestoreV2:8*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@pi/log_std
Ż
save_18/Assign_9Assignvc/dense/biassave_18/RestoreV2:9*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
validate_shape(*
use_locking(
ś
save_18/Assign_10Assignvc/dense/bias/Adamsave_18/RestoreV2:10*
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0*
_output_shapes	
:
¸
save_18/Assign_11Assignvc/dense/bias/Adam_1save_18/RestoreV2:11*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0*
use_locking(
š
save_18/Assign_12Assignvc/dense/kernelsave_18/RestoreV2:12*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<*
validate_shape(*
use_locking(
ž
save_18/Assign_13Assignvc/dense/kernel/Adamsave_18/RestoreV2:13*
_output_shapes
:	<*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel
Ŕ
save_18/Assign_14Assignvc/dense/kernel/Adam_1save_18/RestoreV2:14*
_output_shapes
:	<*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(
ľ
save_18/Assign_15Assignvc/dense_1/biassave_18/RestoreV2:15*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0
ş
save_18/Assign_16Assignvc/dense_1/bias/Adamsave_18/RestoreV2:16*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
use_locking(*
validate_shape(*
T0
ź
save_18/Assign_17Assignvc/dense_1/bias/Adam_1save_18/RestoreV2:17*"
_class
loc:@vc/dense_1/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:
ž
save_18/Assign_18Assignvc/dense_1/kernelsave_18/RestoreV2:18*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0
Ă
save_18/Assign_19Assignvc/dense_1/kernel/Adamsave_18/RestoreV2:19* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
Ĺ
save_18/Assign_20Assignvc/dense_1/kernel/Adam_1save_18/RestoreV2:20*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
´
save_18/Assign_21Assignvc/dense_2/biassave_18/RestoreV2:21*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(
š
save_18/Assign_22Assignvc/dense_2/bias/Adamsave_18/RestoreV2:22*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
ť
save_18/Assign_23Assignvc/dense_2/bias/Adam_1save_18/RestoreV2:23*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
˝
save_18/Assign_24Assignvc/dense_2/kernelsave_18/RestoreV2:24*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
Â
save_18/Assign_25Assignvc/dense_2/kernel/Adamsave_18/RestoreV2:25*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
Ä
save_18/Assign_26Assignvc/dense_2/kernel/Adam_1save_18/RestoreV2:26*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
ą
save_18/Assign_27Assignvf/dense/biassave_18/RestoreV2:27* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(
ś
save_18/Assign_28Assignvf/dense/bias/Adamsave_18/RestoreV2:28*
validate_shape(*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias*
use_locking(
¸
save_18/Assign_29Assignvf/dense/bias/Adam_1save_18/RestoreV2:29*
T0* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:
š
save_18/Assign_30Assignvf/dense/kernelsave_18/RestoreV2:30*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(*
T0
ž
save_18/Assign_31Assignvf/dense/kernel/Adamsave_18/RestoreV2:31*
T0*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(
Ŕ
save_18/Assign_32Assignvf/dense/kernel/Adam_1save_18/RestoreV2:32*
use_locking(*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	<*
validate_shape(
ľ
save_18/Assign_33Assignvf/dense_1/biassave_18/RestoreV2:33*
use_locking(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0*
validate_shape(
ş
save_18/Assign_34Assignvf/dense_1/bias/Adamsave_18/RestoreV2:34*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:
ź
save_18/Assign_35Assignvf/dense_1/bias/Adam_1save_18/RestoreV2:35*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0
ž
save_18/Assign_36Assignvf/dense_1/kernelsave_18/RestoreV2:36*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel
Ă
save_18/Assign_37Assignvf/dense_1/kernel/Adamsave_18/RestoreV2:37* 
_output_shapes
:
*
T0*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(
Ĺ
save_18/Assign_38Assignvf/dense_1/kernel/Adam_1save_18/RestoreV2:38*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:

´
save_18/Assign_39Assignvf/dense_2/biassave_18/RestoreV2:39*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
š
save_18/Assign_40Assignvf/dense_2/bias/Adamsave_18/RestoreV2:40*"
_class
loc:@vf/dense_2/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:
ť
save_18/Assign_41Assignvf/dense_2/bias/Adam_1save_18/RestoreV2:41*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
˝
save_18/Assign_42Assignvf/dense_2/kernelsave_18/RestoreV2:42*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
T0*
validate_shape(
Â
save_18/Assign_43Assignvf/dense_2/kernel/Adamsave_18/RestoreV2:43*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel
Ä
save_18/Assign_44Assignvf/dense_2/kernel/Adam_1save_18/RestoreV2:44*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
T0

save_18/restore_shardNoOp^save_18/Assign^save_18/Assign_1^save_18/Assign_10^save_18/Assign_11^save_18/Assign_12^save_18/Assign_13^save_18/Assign_14^save_18/Assign_15^save_18/Assign_16^save_18/Assign_17^save_18/Assign_18^save_18/Assign_19^save_18/Assign_2^save_18/Assign_20^save_18/Assign_21^save_18/Assign_22^save_18/Assign_23^save_18/Assign_24^save_18/Assign_25^save_18/Assign_26^save_18/Assign_27^save_18/Assign_28^save_18/Assign_29^save_18/Assign_3^save_18/Assign_30^save_18/Assign_31^save_18/Assign_32^save_18/Assign_33^save_18/Assign_34^save_18/Assign_35^save_18/Assign_36^save_18/Assign_37^save_18/Assign_38^save_18/Assign_39^save_18/Assign_4^save_18/Assign_40^save_18/Assign_41^save_18/Assign_42^save_18/Assign_43^save_18/Assign_44^save_18/Assign_5^save_18/Assign_6^save_18/Assign_7^save_18/Assign_8^save_18/Assign_9
3
save_18/restore_allNoOp^save_18/restore_shard
\
save_19/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_19/filenamePlaceholderWithDefaultsave_19/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_19/ConstPlaceholderWithDefaultsave_19/filename*
dtype0*
_output_shapes
: *
shape: 

save_19/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_b9a32b2dd9a449dda857a93b810e3f62/part
~
save_19/StringJoin
StringJoinsave_19/Constsave_19/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_19/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_19/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_19/ShardedFilenameShardedFilenamesave_19/StringJoinsave_19/ShardedFilename/shardsave_19/num_shards*
_output_shapes
: 
ó
save_19/SaveV2/tensor_namesConst*
_output_shapes
:-*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ŕ
save_19/SaveV2/shape_and_slicesConst*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-
ş
save_19/SaveV2SaveV2save_19/ShardedFilenamesave_19/SaveV2/tensor_namessave_19/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_19/control_dependencyIdentitysave_19/ShardedFilename^save_19/SaveV2*
T0**
_class 
loc:@save_19/ShardedFilename*
_output_shapes
: 
Ś
.save_19/MergeV2Checkpoints/checkpoint_prefixesPacksave_19/ShardedFilename^save_19/control_dependency*
T0*
N*
_output_shapes
:*

axis 

save_19/MergeV2CheckpointsMergeV2Checkpoints.save_19/MergeV2Checkpoints/checkpoint_prefixessave_19/Const*
delete_old_dirs(

save_19/IdentityIdentitysave_19/Const^save_19/MergeV2Checkpoints^save_19/control_dependency*
_output_shapes
: *
T0
ö
save_19/RestoreV2/tensor_namesConst*
_output_shapes
:-*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ă
"save_19/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ű
save_19/RestoreV2	RestoreV2save_19/Constsave_19/RestoreV2/tensor_names"save_19/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
¤
save_19/AssignAssignbeta1_powersave_19/RestoreV2*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
T0
¨
save_19/Assign_1Assignbeta2_powersave_19/RestoreV2:1*
_output_shapes
: *
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
T0
Ż
save_19/Assign_2Assignpi/dense/biassave_19/RestoreV2:2*
T0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(
ˇ
save_19/Assign_3Assignpi/dense/kernelsave_19/RestoreV2:3*
use_locking(*
validate_shape(*
_output_shapes
:	<*
T0*"
_class
loc:@pi/dense/kernel
ł
save_19/Assign_4Assignpi/dense_1/biassave_19/RestoreV2:4*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:
ź
save_19/Assign_5Assignpi/dense_1/kernelsave_19/RestoreV2:5* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(*
use_locking(
˛
save_19/Assign_6Assignpi/dense_2/biassave_19/RestoreV2:6*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
ť
save_19/Assign_7Assignpi/dense_2/kernelsave_19/RestoreV2:7*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
T0
¨
save_19/Assign_8Assign
pi/log_stdsave_19/RestoreV2:8*
_class
loc:@pi/log_std*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
Ż
save_19/Assign_9Assignvc/dense/biassave_19/RestoreV2:9*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
ś
save_19/Assign_10Assignvc/dense/bias/Adamsave_19/RestoreV2:10* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
¸
save_19/Assign_11Assignvc/dense/bias/Adam_1save_19/RestoreV2:11*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:
š
save_19/Assign_12Assignvc/dense/kernelsave_19/RestoreV2:12*
validate_shape(*
_output_shapes
:	<*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel
ž
save_19/Assign_13Assignvc/dense/kernel/Adamsave_19/RestoreV2:13*
validate_shape(*
_output_shapes
:	<*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(
Ŕ
save_19/Assign_14Assignvc/dense/kernel/Adam_1save_19/RestoreV2:14*
T0*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
use_locking(
ľ
save_19/Assign_15Assignvc/dense_1/biassave_19/RestoreV2:15*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
ş
save_19/Assign_16Assignvc/dense_1/bias/Adamsave_19/RestoreV2:16*
use_locking(*
_output_shapes	
:*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(
ź
save_19/Assign_17Assignvc/dense_1/bias/Adam_1save_19/RestoreV2:17*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(
ž
save_19/Assign_18Assignvc/dense_1/kernelsave_19/RestoreV2:18*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
Ă
save_19/Assign_19Assignvc/dense_1/kernel/Adamsave_19/RestoreV2:19*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:

Ĺ
save_19/Assign_20Assignvc/dense_1/kernel/Adam_1save_19/RestoreV2:20*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(*
use_locking(* 
_output_shapes
:

´
save_19/Assign_21Assignvc/dense_2/biassave_19/RestoreV2:21*"
_class
loc:@vc/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
š
save_19/Assign_22Assignvc/dense_2/bias/Adamsave_19/RestoreV2:22*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:
ť
save_19/Assign_23Assignvc/dense_2/bias/Adam_1save_19/RestoreV2:23*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0
˝
save_19/Assign_24Assignvc/dense_2/kernelsave_19/RestoreV2:24*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0*
use_locking(
Â
save_19/Assign_25Assignvc/dense_2/kernel/Adamsave_19/RestoreV2:25*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	
Ä
save_19/Assign_26Assignvc/dense_2/kernel/Adam_1save_19/RestoreV2:26*
validate_shape(*
T0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
use_locking(
ą
save_19/Assign_27Assignvf/dense/biassave_19/RestoreV2:27*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vf/dense/bias*
use_locking(
ś
save_19/Assign_28Assignvf/dense/bias/Adamsave_19/RestoreV2:28*
T0* 
_class
loc:@vf/dense/bias*
use_locking(*
_output_shapes	
:*
validate_shape(
¸
save_19/Assign_29Assignvf/dense/bias/Adam_1save_19/RestoreV2:29*
T0* 
_class
loc:@vf/dense/bias*
use_locking(*
_output_shapes	
:*
validate_shape(
š
save_19/Assign_30Assignvf/dense/kernelsave_19/RestoreV2:30*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<
ž
save_19/Assign_31Assignvf/dense/kernel/Adamsave_19/RestoreV2:31*
T0*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(
Ŕ
save_19/Assign_32Assignvf/dense/kernel/Adam_1save_19/RestoreV2:32*
T0*
validate_shape(*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	<
ľ
save_19/Assign_33Assignvf/dense_1/biassave_19/RestoreV2:33*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias
ş
save_19/Assign_34Assignvf/dense_1/bias/Adamsave_19/RestoreV2:34*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:
ź
save_19/Assign_35Assignvf/dense_1/bias/Adam_1save_19/RestoreV2:35*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0*
validate_shape(*
use_locking(
ž
save_19/Assign_36Assignvf/dense_1/kernelsave_19/RestoreV2:36*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:

Ă
save_19/Assign_37Assignvf/dense_1/kernel/Adamsave_19/RestoreV2:37*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:
*
T0
Ĺ
save_19/Assign_38Assignvf/dense_1/kernel/Adam_1save_19/RestoreV2:38*
T0*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:

´
save_19/Assign_39Assignvf/dense_2/biassave_19/RestoreV2:39*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(
š
save_19/Assign_40Assignvf/dense_2/bias/Adamsave_19/RestoreV2:40*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
ť
save_19/Assign_41Assignvf/dense_2/bias/Adam_1save_19/RestoreV2:41*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0*
use_locking(
˝
save_19/Assign_42Assignvf/dense_2/kernelsave_19/RestoreV2:42*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_19/Assign_43Assignvf/dense_2/kernel/Adamsave_19/RestoreV2:43*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	
Ä
save_19/Assign_44Assignvf/dense_2/kernel/Adam_1save_19/RestoreV2:44*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	

save_19/restore_shardNoOp^save_19/Assign^save_19/Assign_1^save_19/Assign_10^save_19/Assign_11^save_19/Assign_12^save_19/Assign_13^save_19/Assign_14^save_19/Assign_15^save_19/Assign_16^save_19/Assign_17^save_19/Assign_18^save_19/Assign_19^save_19/Assign_2^save_19/Assign_20^save_19/Assign_21^save_19/Assign_22^save_19/Assign_23^save_19/Assign_24^save_19/Assign_25^save_19/Assign_26^save_19/Assign_27^save_19/Assign_28^save_19/Assign_29^save_19/Assign_3^save_19/Assign_30^save_19/Assign_31^save_19/Assign_32^save_19/Assign_33^save_19/Assign_34^save_19/Assign_35^save_19/Assign_36^save_19/Assign_37^save_19/Assign_38^save_19/Assign_39^save_19/Assign_4^save_19/Assign_40^save_19/Assign_41^save_19/Assign_42^save_19/Assign_43^save_19/Assign_44^save_19/Assign_5^save_19/Assign_6^save_19/Assign_7^save_19/Assign_8^save_19/Assign_9
3
save_19/restore_allNoOp^save_19/restore_shard
\
save_20/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_20/filenamePlaceholderWithDefaultsave_20/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_20/ConstPlaceholderWithDefaultsave_20/filename*
_output_shapes
: *
shape: *
dtype0

save_20/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_e36037a306b54c37b426b996de9e56be/part
~
save_20/StringJoin
StringJoinsave_20/Constsave_20/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_20/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_20/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_20/ShardedFilenameShardedFilenamesave_20/StringJoinsave_20/ShardedFilename/shardsave_20/num_shards*
_output_shapes
: 
ó
save_20/SaveV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
Ŕ
save_20/SaveV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ş
save_20/SaveV2SaveV2save_20/ShardedFilenamesave_20/SaveV2/tensor_namessave_20/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_20/control_dependencyIdentitysave_20/ShardedFilename^save_20/SaveV2*
_output_shapes
: **
_class 
loc:@save_20/ShardedFilename*
T0
Ś
.save_20/MergeV2Checkpoints/checkpoint_prefixesPacksave_20/ShardedFilename^save_20/control_dependency*
_output_shapes
:*

axis *
N*
T0

save_20/MergeV2CheckpointsMergeV2Checkpoints.save_20/MergeV2Checkpoints/checkpoint_prefixessave_20/Const*
delete_old_dirs(

save_20/IdentityIdentitysave_20/Const^save_20/MergeV2Checkpoints^save_20/control_dependency*
T0*
_output_shapes
: 
ö
save_20/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
Ă
"save_20/RestoreV2/shape_and_slicesConst*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-
ű
save_20/RestoreV2	RestoreV2save_20/Constsave_20/RestoreV2/tensor_names"save_20/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
¤
save_20/AssignAssignbeta1_powersave_20/RestoreV2*
T0*
_output_shapes
: *
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(
¨
save_20/Assign_1Assignbeta2_powersave_20/RestoreV2:1*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0
Ż
save_20/Assign_2Assignpi/dense/biassave_20/RestoreV2:2*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
use_locking(
ˇ
save_20/Assign_3Assignpi/dense/kernelsave_20/RestoreV2:3*
use_locking(*
_output_shapes
:	<*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel
ł
save_20/Assign_4Assignpi/dense_1/biassave_20/RestoreV2:4*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:
ź
save_20/Assign_5Assignpi/dense_1/kernelsave_20/RestoreV2:5* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
˛
save_20/Assign_6Assignpi/dense_2/biassave_20/RestoreV2:6*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
ť
save_20/Assign_7Assignpi/dense_2/kernelsave_20/RestoreV2:7*
T0*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel
¨
save_20/Assign_8Assign
pi/log_stdsave_20/RestoreV2:8*
_output_shapes
:*
validate_shape(*
T0*
_class
loc:@pi/log_std*
use_locking(
Ż
save_20/Assign_9Assignvc/dense/biassave_20/RestoreV2:9* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
ś
save_20/Assign_10Assignvc/dense/bias/Adamsave_20/RestoreV2:10*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0
¸
save_20/Assign_11Assignvc/dense/bias/Adam_1save_20/RestoreV2:11*
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:*
T0
š
save_20/Assign_12Assignvc/dense/kernelsave_20/RestoreV2:12*
use_locking(*
_output_shapes
:	<*
T0*
validate_shape(*"
_class
loc:@vc/dense/kernel
ž
save_20/Assign_13Assignvc/dense/kernel/Adamsave_20/RestoreV2:13*
use_locking(*
_output_shapes
:	<*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(
Ŕ
save_20/Assign_14Assignvc/dense/kernel/Adam_1save_20/RestoreV2:14*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0
ľ
save_20/Assign_15Assignvc/dense_1/biassave_20/RestoreV2:15*"
_class
loc:@vc/dense_1/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
ş
save_20/Assign_16Assignvc/dense_1/bias/Adamsave_20/RestoreV2:16*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0
ź
save_20/Assign_17Assignvc/dense_1/bias/Adam_1save_20/RestoreV2:17*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(
ž
save_20/Assign_18Assignvc/dense_1/kernelsave_20/RestoreV2:18*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:

Ă
save_20/Assign_19Assignvc/dense_1/kernel/Adamsave_20/RestoreV2:19*
T0*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
validate_shape(
Ĺ
save_20/Assign_20Assignvc/dense_1/kernel/Adam_1save_20/RestoreV2:20*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
T0
´
save_20/Assign_21Assignvc/dense_2/biassave_20/RestoreV2:21*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(
š
save_20/Assign_22Assignvc/dense_2/bias/Adamsave_20/RestoreV2:22*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_20/Assign_23Assignvc/dense_2/bias/Adam_1save_20/RestoreV2:23*
T0*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(
˝
save_20/Assign_24Assignvc/dense_2/kernelsave_20/RestoreV2:24*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	
Â
save_20/Assign_25Assignvc/dense_2/kernel/Adamsave_20/RestoreV2:25*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(
Ä
save_20/Assign_26Assignvc/dense_2/kernel/Adam_1save_20/RestoreV2:26*
validate_shape(*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0
ą
save_20/Assign_27Assignvf/dense/biassave_20/RestoreV2:27*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(* 
_class
loc:@vf/dense/bias
ś
save_20/Assign_28Assignvf/dense/bias/Adamsave_20/RestoreV2:28*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
¸
save_20/Assign_29Assignvf/dense/bias/Adam_1save_20/RestoreV2:29* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
š
save_20/Assign_30Assignvf/dense/kernelsave_20/RestoreV2:30*
use_locking(*
T0*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@vf/dense/kernel
ž
save_20/Assign_31Assignvf/dense/kernel/Adamsave_20/RestoreV2:31*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<
Ŕ
save_20/Assign_32Assignvf/dense/kernel/Adam_1save_20/RestoreV2:32*
validate_shape(*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	<*
use_locking(
ľ
save_20/Assign_33Assignvf/dense_1/biassave_20/RestoreV2:33*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
ş
save_20/Assign_34Assignvf/dense_1/bias/Adamsave_20/RestoreV2:34*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:
ź
save_20/Assign_35Assignvf/dense_1/bias/Adam_1save_20/RestoreV2:35*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_1/bias
ž
save_20/Assign_36Assignvf/dense_1/kernelsave_20/RestoreV2:36*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
Ă
save_20/Assign_37Assignvf/dense_1/kernel/Adamsave_20/RestoreV2:37*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
Ĺ
save_20/Assign_38Assignvf/dense_1/kernel/Adam_1save_20/RestoreV2:38*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0
´
save_20/Assign_39Assignvf/dense_2/biassave_20/RestoreV2:39*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
š
save_20/Assign_40Assignvf/dense_2/bias/Adamsave_20/RestoreV2:40*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
ť
save_20/Assign_41Assignvf/dense_2/bias/Adam_1save_20/RestoreV2:41*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
˝
save_20/Assign_42Assignvf/dense_2/kernelsave_20/RestoreV2:42*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(
Â
save_20/Assign_43Assignvf/dense_2/kernel/Adamsave_20/RestoreV2:43*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0
Ä
save_20/Assign_44Assignvf/dense_2/kernel/Adam_1save_20/RestoreV2:44*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(

save_20/restore_shardNoOp^save_20/Assign^save_20/Assign_1^save_20/Assign_10^save_20/Assign_11^save_20/Assign_12^save_20/Assign_13^save_20/Assign_14^save_20/Assign_15^save_20/Assign_16^save_20/Assign_17^save_20/Assign_18^save_20/Assign_19^save_20/Assign_2^save_20/Assign_20^save_20/Assign_21^save_20/Assign_22^save_20/Assign_23^save_20/Assign_24^save_20/Assign_25^save_20/Assign_26^save_20/Assign_27^save_20/Assign_28^save_20/Assign_29^save_20/Assign_3^save_20/Assign_30^save_20/Assign_31^save_20/Assign_32^save_20/Assign_33^save_20/Assign_34^save_20/Assign_35^save_20/Assign_36^save_20/Assign_37^save_20/Assign_38^save_20/Assign_39^save_20/Assign_4^save_20/Assign_40^save_20/Assign_41^save_20/Assign_42^save_20/Assign_43^save_20/Assign_44^save_20/Assign_5^save_20/Assign_6^save_20/Assign_7^save_20/Assign_8^save_20/Assign_9
3
save_20/restore_allNoOp^save_20/restore_shard
\
save_21/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_21/filenamePlaceholderWithDefaultsave_21/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_21/ConstPlaceholderWithDefaultsave_21/filename*
shape: *
dtype0*
_output_shapes
: 

save_21/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_b24189960362465cbc75fec9986a0a0a/part*
_output_shapes
: 
~
save_21/StringJoin
StringJoinsave_21/Constsave_21/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_21/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_21/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_21/ShardedFilenameShardedFilenamesave_21/StringJoinsave_21/ShardedFilename/shardsave_21/num_shards*
_output_shapes
: 
ó
save_21/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ŕ
save_21/SaveV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ş
save_21/SaveV2SaveV2save_21/ShardedFilenamesave_21/SaveV2/tensor_namessave_21/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_21/control_dependencyIdentitysave_21/ShardedFilename^save_21/SaveV2**
_class 
loc:@save_21/ShardedFilename*
T0*
_output_shapes
: 
Ś
.save_21/MergeV2Checkpoints/checkpoint_prefixesPacksave_21/ShardedFilename^save_21/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_21/MergeV2CheckpointsMergeV2Checkpoints.save_21/MergeV2Checkpoints/checkpoint_prefixessave_21/Const*
delete_old_dirs(

save_21/IdentityIdentitysave_21/Const^save_21/MergeV2Checkpoints^save_21/control_dependency*
T0*
_output_shapes
: 
ö
save_21/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
Ă
"save_21/RestoreV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-*
dtype0
ű
save_21/RestoreV2	RestoreV2save_21/Constsave_21/RestoreV2/tensor_names"save_21/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
¤
save_21/AssignAssignbeta1_powersave_21/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@vc/dense/bias
¨
save_21/Assign_1Assignbeta2_powersave_21/RestoreV2:1*
T0*
validate_shape(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
use_locking(
Ż
save_21/Assign_2Assignpi/dense/biassave_21/RestoreV2:2* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(
ˇ
save_21/Assign_3Assignpi/dense/kernelsave_21/RestoreV2:3*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
use_locking(*
validate_shape(
ł
save_21/Assign_4Assignpi/dense_1/biassave_21/RestoreV2:4*
use_locking(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0
ź
save_21/Assign_5Assignpi/dense_1/kernelsave_21/RestoreV2:5*
use_locking(*
T0* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
˛
save_21/Assign_6Assignpi/dense_2/biassave_21/RestoreV2:6*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias
ť
save_21/Assign_7Assignpi/dense_2/kernelsave_21/RestoreV2:7*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel
¨
save_21/Assign_8Assign
pi/log_stdsave_21/RestoreV2:8*
T0*
use_locking(*
_output_shapes
:*
_class
loc:@pi/log_std*
validate_shape(
Ż
save_21/Assign_9Assignvc/dense/biassave_21/RestoreV2:9*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
validate_shape(
ś
save_21/Assign_10Assignvc/dense/bias/Adamsave_21/RestoreV2:10*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
¸
save_21/Assign_11Assignvc/dense/bias/Adam_1save_21/RestoreV2:11*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(
š
save_21/Assign_12Assignvc/dense/kernelsave_21/RestoreV2:12*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	<*
use_locking(
ž
save_21/Assign_13Assignvc/dense/kernel/Adamsave_21/RestoreV2:13*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<*
use_locking(*
validate_shape(
Ŕ
save_21/Assign_14Assignvc/dense/kernel/Adam_1save_21/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	<*
T0
ľ
save_21/Assign_15Assignvc/dense_1/biassave_21/RestoreV2:15*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0*
validate_shape(
ş
save_21/Assign_16Assignvc/dense_1/bias/Adamsave_21/RestoreV2:16*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vc/dense_1/bias
ź
save_21/Assign_17Assignvc/dense_1/bias/Adam_1save_21/RestoreV2:17*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
ž
save_21/Assign_18Assignvc/dense_1/kernelsave_21/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:
*
validate_shape(
Ă
save_21/Assign_19Assignvc/dense_1/kernel/Adamsave_21/RestoreV2:19*
validate_shape(*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
T0
Ĺ
save_21/Assign_20Assignvc/dense_1/kernel/Adam_1save_21/RestoreV2:20*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
´
save_21/Assign_21Assignvc/dense_2/biassave_21/RestoreV2:21*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
š
save_21/Assign_22Assignvc/dense_2/bias/Adamsave_21/RestoreV2:22*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(
ť
save_21/Assign_23Assignvc/dense_2/bias/Adam_1save_21/RestoreV2:23*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense_2/bias
˝
save_21/Assign_24Assignvc/dense_2/kernelsave_21/RestoreV2:24*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(*
T0
Â
save_21/Assign_25Assignvc/dense_2/kernel/Adamsave_21/RestoreV2:25*
validate_shape(*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(
Ä
save_21/Assign_26Assignvc/dense_2/kernel/Adam_1save_21/RestoreV2:26*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
ą
save_21/Assign_27Assignvf/dense/biassave_21/RestoreV2:27*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
T0
ś
save_21/Assign_28Assignvf/dense/bias/Adamsave_21/RestoreV2:28*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0* 
_class
loc:@vf/dense/bias
¸
save_21/Assign_29Assignvf/dense/bias/Adam_1save_21/RestoreV2:29*
use_locking(* 
_class
loc:@vf/dense/bias*
T0*
validate_shape(*
_output_shapes	
:
š
save_21/Assign_30Assignvf/dense/kernelsave_21/RestoreV2:30*"
_class
loc:@vf/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	<
ž
save_21/Assign_31Assignvf/dense/kernel/Adamsave_21/RestoreV2:31*"
_class
loc:@vf/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	<
Ŕ
save_21/Assign_32Assignvf/dense/kernel/Adam_1save_21/RestoreV2:32*
T0*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
use_locking(
ľ
save_21/Assign_33Assignvf/dense_1/biassave_21/RestoreV2:33*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0*
validate_shape(
ş
save_21/Assign_34Assignvf/dense_1/bias/Adamsave_21/RestoreV2:34*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
ź
save_21/Assign_35Assignvf/dense_1/bias/Adam_1save_21/RestoreV2:35*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(
ž
save_21/Assign_36Assignvf/dense_1/kernelsave_21/RestoreV2:36*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:

Ă
save_21/Assign_37Assignvf/dense_1/kernel/Adamsave_21/RestoreV2:37*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
validate_shape(*
T0
Ĺ
save_21/Assign_38Assignvf/dense_1/kernel/Adam_1save_21/RestoreV2:38*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:
*
use_locking(
´
save_21/Assign_39Assignvf/dense_2/biassave_21/RestoreV2:39*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
validate_shape(
š
save_21/Assign_40Assignvf/dense_2/bias/Adamsave_21/RestoreV2:40*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_21/Assign_41Assignvf/dense_2/bias/Adam_1save_21/RestoreV2:41*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias
˝
save_21/Assign_42Assignvf/dense_2/kernelsave_21/RestoreV2:42*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
validate_shape(
Â
save_21/Assign_43Assignvf/dense_2/kernel/Adamsave_21/RestoreV2:43*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	
Ä
save_21/Assign_44Assignvf/dense_2/kernel/Adam_1save_21/RestoreV2:44*
_output_shapes
:	*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
use_locking(

save_21/restore_shardNoOp^save_21/Assign^save_21/Assign_1^save_21/Assign_10^save_21/Assign_11^save_21/Assign_12^save_21/Assign_13^save_21/Assign_14^save_21/Assign_15^save_21/Assign_16^save_21/Assign_17^save_21/Assign_18^save_21/Assign_19^save_21/Assign_2^save_21/Assign_20^save_21/Assign_21^save_21/Assign_22^save_21/Assign_23^save_21/Assign_24^save_21/Assign_25^save_21/Assign_26^save_21/Assign_27^save_21/Assign_28^save_21/Assign_29^save_21/Assign_3^save_21/Assign_30^save_21/Assign_31^save_21/Assign_32^save_21/Assign_33^save_21/Assign_34^save_21/Assign_35^save_21/Assign_36^save_21/Assign_37^save_21/Assign_38^save_21/Assign_39^save_21/Assign_4^save_21/Assign_40^save_21/Assign_41^save_21/Assign_42^save_21/Assign_43^save_21/Assign_44^save_21/Assign_5^save_21/Assign_6^save_21/Assign_7^save_21/Assign_8^save_21/Assign_9
3
save_21/restore_allNoOp^save_21/restore_shard
\
save_22/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_22/filenamePlaceholderWithDefaultsave_22/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_22/ConstPlaceholderWithDefaultsave_22/filename*
dtype0*
shape: *
_output_shapes
: 

save_22/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_29d6536d604d49e2b36a952bb6e86bc7/part*
dtype0
~
save_22/StringJoin
StringJoinsave_22/Constsave_22/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
T
save_22/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_22/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_22/ShardedFilenameShardedFilenamesave_22/StringJoinsave_22/ShardedFilename/shardsave_22/num_shards*
_output_shapes
: 
ó
save_22/SaveV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
Ŕ
save_22/SaveV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
ş
save_22/SaveV2SaveV2save_22/ShardedFilenamesave_22/SaveV2/tensor_namessave_22/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_22/control_dependencyIdentitysave_22/ShardedFilename^save_22/SaveV2**
_class 
loc:@save_22/ShardedFilename*
_output_shapes
: *
T0
Ś
.save_22/MergeV2Checkpoints/checkpoint_prefixesPacksave_22/ShardedFilename^save_22/control_dependency*
T0*
_output_shapes
:*

axis *
N

save_22/MergeV2CheckpointsMergeV2Checkpoints.save_22/MergeV2Checkpoints/checkpoint_prefixessave_22/Const*
delete_old_dirs(

save_22/IdentityIdentitysave_22/Const^save_22/MergeV2Checkpoints^save_22/control_dependency*
_output_shapes
: *
T0
ö
save_22/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ă
"save_22/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ű
save_22/RestoreV2	RestoreV2save_22/Constsave_22/RestoreV2/tensor_names"save_22/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
¤
save_22/AssignAssignbeta1_powersave_22/RestoreV2* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
_output_shapes
: *
validate_shape(
¨
save_22/Assign_1Assignbeta2_powersave_22/RestoreV2:1* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
Ż
save_22/Assign_2Assignpi/dense/biassave_22/RestoreV2:2*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes	
:*
validate_shape(
ˇ
save_22/Assign_3Assignpi/dense/kernelsave_22/RestoreV2:3*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel
ł
save_22/Assign_4Assignpi/dense_1/biassave_22/RestoreV2:4*
_output_shapes	
:*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(
ź
save_22/Assign_5Assignpi/dense_1/kernelsave_22/RestoreV2:5*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:

˛
save_22/Assign_6Assignpi/dense_2/biassave_22/RestoreV2:6*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
T0
ť
save_22/Assign_7Assignpi/dense_2/kernelsave_22/RestoreV2:7*
T0*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(
¨
save_22/Assign_8Assign
pi/log_stdsave_22/RestoreV2:8*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*
_class
loc:@pi/log_std
Ż
save_22/Assign_9Assignvc/dense/biassave_22/RestoreV2:9*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@vc/dense/bias
ś
save_22/Assign_10Assignvc/dense/bias/Adamsave_22/RestoreV2:10*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0*
use_locking(
¸
save_22/Assign_11Assignvc/dense/bias/Adam_1save_22/RestoreV2:11*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
use_locking(*
T0*
validate_shape(
š
save_22/Assign_12Assignvc/dense/kernelsave_22/RestoreV2:12*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
validate_shape(*
use_locking(*
T0
ž
save_22/Assign_13Assignvc/dense/kernel/Adamsave_22/RestoreV2:13*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	<
Ŕ
save_22/Assign_14Assignvc/dense/kernel/Adam_1save_22/RestoreV2:14*
T0*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
use_locking(
ľ
save_22/Assign_15Assignvc/dense_1/biassave_22/RestoreV2:15*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0
ş
save_22/Assign_16Assignvc/dense_1/bias/Adamsave_22/RestoreV2:16*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
ź
save_22/Assign_17Assignvc/dense_1/bias/Adam_1save_22/RestoreV2:17*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
T0*
validate_shape(*
use_locking(
ž
save_22/Assign_18Assignvc/dense_1/kernelsave_22/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:

Ă
save_22/Assign_19Assignvc/dense_1/kernel/Adamsave_22/RestoreV2:19*
T0*
use_locking(* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
Ĺ
save_22/Assign_20Assignvc/dense_1/kernel/Adam_1save_22/RestoreV2:20*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:

´
save_22/Assign_21Assignvc/dense_2/biassave_22/RestoreV2:21*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
š
save_22/Assign_22Assignvc/dense_2/bias/Adamsave_22/RestoreV2:22*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
ť
save_22/Assign_23Assignvc/dense_2/bias/Adam_1save_22/RestoreV2:23*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
˝
save_22/Assign_24Assignvc/dense_2/kernelsave_22/RestoreV2:24*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	
Â
save_22/Assign_25Assignvc/dense_2/kernel/Adamsave_22/RestoreV2:25*
_output_shapes
:	*
validate_shape(*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(
Ä
save_22/Assign_26Assignvc/dense_2/kernel/Adam_1save_22/RestoreV2:26*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel
ą
save_22/Assign_27Assignvf/dense/biassave_22/RestoreV2:27* 
_class
loc:@vf/dense/bias*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(
ś
save_22/Assign_28Assignvf/dense/bias/Adamsave_22/RestoreV2:28*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(
¸
save_22/Assign_29Assignvf/dense/bias/Adam_1save_22/RestoreV2:29*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(*
_output_shapes	
:*
T0
š
save_22/Assign_30Assignvf/dense/kernelsave_22/RestoreV2:30*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
validate_shape(*
use_locking(*
T0
ž
save_22/Assign_31Assignvf/dense/kernel/Adamsave_22/RestoreV2:31*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
use_locking(
Ŕ
save_22/Assign_32Assignvf/dense/kernel/Adam_1save_22/RestoreV2:32*
_output_shapes
:	<*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense/kernel
ľ
save_22/Assign_33Assignvf/dense_1/biassave_22/RestoreV2:33*
use_locking(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0
ş
save_22/Assign_34Assignvf/dense_1/bias/Adamsave_22/RestoreV2:34*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(
ź
save_22/Assign_35Assignvf/dense_1/bias/Adam_1save_22/RestoreV2:35*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
ž
save_22/Assign_36Assignvf/dense_1/kernelsave_22/RestoreV2:36*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
use_locking(
Ă
save_22/Assign_37Assignvf/dense_1/kernel/Adamsave_22/RestoreV2:37* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(
Ĺ
save_22/Assign_38Assignvf/dense_1/kernel/Adam_1save_22/RestoreV2:38* 
_output_shapes
:
*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel*
use_locking(
´
save_22/Assign_39Assignvf/dense_2/biassave_22/RestoreV2:39*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
T0
š
save_22/Assign_40Assignvf/dense_2/bias/Adamsave_22/RestoreV2:40*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
ť
save_22/Assign_41Assignvf/dense_2/bias/Adam_1save_22/RestoreV2:41*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(*
T0*
_output_shapes
:
˝
save_22/Assign_42Assignvf/dense_2/kernelsave_22/RestoreV2:42*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel
Â
save_22/Assign_43Assignvf/dense_2/kernel/Adamsave_22/RestoreV2:43*
validate_shape(*
_output_shapes
:	*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(
Ä
save_22/Assign_44Assignvf/dense_2/kernel/Adam_1save_22/RestoreV2:44*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0

save_22/restore_shardNoOp^save_22/Assign^save_22/Assign_1^save_22/Assign_10^save_22/Assign_11^save_22/Assign_12^save_22/Assign_13^save_22/Assign_14^save_22/Assign_15^save_22/Assign_16^save_22/Assign_17^save_22/Assign_18^save_22/Assign_19^save_22/Assign_2^save_22/Assign_20^save_22/Assign_21^save_22/Assign_22^save_22/Assign_23^save_22/Assign_24^save_22/Assign_25^save_22/Assign_26^save_22/Assign_27^save_22/Assign_28^save_22/Assign_29^save_22/Assign_3^save_22/Assign_30^save_22/Assign_31^save_22/Assign_32^save_22/Assign_33^save_22/Assign_34^save_22/Assign_35^save_22/Assign_36^save_22/Assign_37^save_22/Assign_38^save_22/Assign_39^save_22/Assign_4^save_22/Assign_40^save_22/Assign_41^save_22/Assign_42^save_22/Assign_43^save_22/Assign_44^save_22/Assign_5^save_22/Assign_6^save_22/Assign_7^save_22/Assign_8^save_22/Assign_9
3
save_22/restore_allNoOp^save_22/restore_shard
\
save_23/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_23/filenamePlaceholderWithDefaultsave_23/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_23/ConstPlaceholderWithDefaultsave_23/filename*
dtype0*
shape: *
_output_shapes
: 

save_23/StringJoin/inputs_1Const*<
value3B1 B+_temp_44fac7024a934e498352f797c7da7df9/part*
dtype0*
_output_shapes
: 
~
save_23/StringJoin
StringJoinsave_23/Constsave_23/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_23/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_23/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_23/ShardedFilenameShardedFilenamesave_23/StringJoinsave_23/ShardedFilename/shardsave_23/num_shards*
_output_shapes
: 
ó
save_23/SaveV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
Ŕ
save_23/SaveV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-*
dtype0
ş
save_23/SaveV2SaveV2save_23/ShardedFilenamesave_23/SaveV2/tensor_namessave_23/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_23/control_dependencyIdentitysave_23/ShardedFilename^save_23/SaveV2**
_class 
loc:@save_23/ShardedFilename*
T0*
_output_shapes
: 
Ś
.save_23/MergeV2Checkpoints/checkpoint_prefixesPacksave_23/ShardedFilename^save_23/control_dependency*
N*
T0*
_output_shapes
:*

axis 

save_23/MergeV2CheckpointsMergeV2Checkpoints.save_23/MergeV2Checkpoints/checkpoint_prefixessave_23/Const*
delete_old_dirs(

save_23/IdentityIdentitysave_23/Const^save_23/MergeV2Checkpoints^save_23/control_dependency*
_output_shapes
: *
T0
ö
save_23/RestoreV2/tensor_namesConst*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
Ă
"save_23/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ű
save_23/RestoreV2	RestoreV2save_23/Constsave_23/RestoreV2/tensor_names"save_23/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
¤
save_23/AssignAssignbeta1_powersave_23/RestoreV2* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
: 
¨
save_23/Assign_1Assignbeta2_powersave_23/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias
Ż
save_23/Assign_2Assignpi/dense/biassave_23/RestoreV2:2*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0
ˇ
save_23/Assign_3Assignpi/dense/kernelsave_23/RestoreV2:3*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<
ł
save_23/Assign_4Assignpi/dense_1/biassave_23/RestoreV2:4*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ź
save_23/Assign_5Assignpi/dense_1/kernelsave_23/RestoreV2:5*
T0*
validate_shape(* 
_output_shapes
:
*
use_locking(*$
_class
loc:@pi/dense_1/kernel
˛
save_23/Assign_6Assignpi/dense_2/biassave_23/RestoreV2:6*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(
ť
save_23/Assign_7Assignpi/dense_2/kernelsave_23/RestoreV2:7*
_output_shapes
:	*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(
¨
save_23/Assign_8Assign
pi/log_stdsave_23/RestoreV2:8*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*
_class
loc:@pi/log_std
Ż
save_23/Assign_9Assignvc/dense/biassave_23/RestoreV2:9*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
use_locking(
ś
save_23/Assign_10Assignvc/dense/bias/Adamsave_23/RestoreV2:10*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias
¸
save_23/Assign_11Assignvc/dense/bias/Adam_1save_23/RestoreV2:11*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0*
validate_shape(
š
save_23/Assign_12Assignvc/dense/kernelsave_23/RestoreV2:12*
_output_shapes
:	<*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(
ž
save_23/Assign_13Assignvc/dense/kernel/Adamsave_23/RestoreV2:13*
use_locking(*
T0*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@vc/dense/kernel
Ŕ
save_23/Assign_14Assignvc/dense/kernel/Adam_1save_23/RestoreV2:14*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	<
ľ
save_23/Assign_15Assignvc/dense_1/biassave_23/RestoreV2:15*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(
ş
save_23/Assign_16Assignvc/dense_1/bias/Adamsave_23/RestoreV2:16*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0*"
_class
loc:@vc/dense_1/bias
ź
save_23/Assign_17Assignvc/dense_1/bias/Adam_1save_23/RestoreV2:17*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
use_locking(*
validate_shape(
ž
save_23/Assign_18Assignvc/dense_1/kernelsave_23/RestoreV2:18*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:

Ă
save_23/Assign_19Assignvc/dense_1/kernel/Adamsave_23/RestoreV2:19*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:
*
use_locking(
Ĺ
save_23/Assign_20Assignvc/dense_1/kernel/Adam_1save_23/RestoreV2:20*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0
´
save_23/Assign_21Assignvc/dense_2/biassave_23/RestoreV2:21*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
š
save_23/Assign_22Assignvc/dense_2/bias/Adamsave_23/RestoreV2:22*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
ť
save_23/Assign_23Assignvc/dense_2/bias/Adam_1save_23/RestoreV2:23*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
˝
save_23/Assign_24Assignvc/dense_2/kernelsave_23/RestoreV2:24*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(
Â
save_23/Assign_25Assignvc/dense_2/kernel/Adamsave_23/RestoreV2:25*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
Ä
save_23/Assign_26Assignvc/dense_2/kernel/Adam_1save_23/RestoreV2:26*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(*
T0
ą
save_23/Assign_27Assignvf/dense/biassave_23/RestoreV2:27*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
ś
save_23/Assign_28Assignvf/dense/bias/Adamsave_23/RestoreV2:28* 
_class
loc:@vf/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:
¸
save_23/Assign_29Assignvf/dense/bias/Adam_1save_23/RestoreV2:29*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
T0*
use_locking(
š
save_23/Assign_30Assignvf/dense/kernelsave_23/RestoreV2:30*
T0*
validate_shape(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
use_locking(
ž
save_23/Assign_31Assignvf/dense/kernel/Adamsave_23/RestoreV2:31*
T0*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
use_locking(
Ŕ
save_23/Assign_32Assignvf/dense/kernel/Adam_1save_23/RestoreV2:32*
T0*
use_locking(*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
validate_shape(
ľ
save_23/Assign_33Assignvf/dense_1/biassave_23/RestoreV2:33*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:
ş
save_23/Assign_34Assignvf/dense_1/bias/Adamsave_23/RestoreV2:34*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
use_locking(
ź
save_23/Assign_35Assignvf/dense_1/bias/Adam_1save_23/RestoreV2:35*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
ž
save_23/Assign_36Assignvf/dense_1/kernelsave_23/RestoreV2:36*
validate_shape(*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0
Ă
save_23/Assign_37Assignvf/dense_1/kernel/Adamsave_23/RestoreV2:37*
validate_shape(*
T0* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vf/dense_1/kernel
Ĺ
save_23/Assign_38Assignvf/dense_1/kernel/Adam_1save_23/RestoreV2:38*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
´
save_23/Assign_39Assignvf/dense_2/biassave_23/RestoreV2:39*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
š
save_23/Assign_40Assignvf/dense_2/bias/Adamsave_23/RestoreV2:40*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(
ť
save_23/Assign_41Assignvf/dense_2/bias/Adam_1save_23/RestoreV2:41*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias
˝
save_23/Assign_42Assignvf/dense_2/kernelsave_23/RestoreV2:42*
_output_shapes
:	*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(
Â
save_23/Assign_43Assignvf/dense_2/kernel/Adamsave_23/RestoreV2:43*
use_locking(*
_output_shapes
:	*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
Ä
save_23/Assign_44Assignvf/dense_2/kernel/Adam_1save_23/RestoreV2:44*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
T0

save_23/restore_shardNoOp^save_23/Assign^save_23/Assign_1^save_23/Assign_10^save_23/Assign_11^save_23/Assign_12^save_23/Assign_13^save_23/Assign_14^save_23/Assign_15^save_23/Assign_16^save_23/Assign_17^save_23/Assign_18^save_23/Assign_19^save_23/Assign_2^save_23/Assign_20^save_23/Assign_21^save_23/Assign_22^save_23/Assign_23^save_23/Assign_24^save_23/Assign_25^save_23/Assign_26^save_23/Assign_27^save_23/Assign_28^save_23/Assign_29^save_23/Assign_3^save_23/Assign_30^save_23/Assign_31^save_23/Assign_32^save_23/Assign_33^save_23/Assign_34^save_23/Assign_35^save_23/Assign_36^save_23/Assign_37^save_23/Assign_38^save_23/Assign_39^save_23/Assign_4^save_23/Assign_40^save_23/Assign_41^save_23/Assign_42^save_23/Assign_43^save_23/Assign_44^save_23/Assign_5^save_23/Assign_6^save_23/Assign_7^save_23/Assign_8^save_23/Assign_9
3
save_23/restore_allNoOp^save_23/restore_shard
\
save_24/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_24/filenamePlaceholderWithDefaultsave_24/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_24/ConstPlaceholderWithDefaultsave_24/filename*
dtype0*
_output_shapes
: *
shape: 

save_24/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_e882c29baf9e4f79a1dcce1f168605b1/part*
_output_shapes
: 
~
save_24/StringJoin
StringJoinsave_24/Constsave_24/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_24/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
_
save_24/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 

save_24/ShardedFilenameShardedFilenamesave_24/StringJoinsave_24/ShardedFilename/shardsave_24/num_shards*
_output_shapes
: 
ó
save_24/SaveV2/tensor_namesConst*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-
Ŕ
save_24/SaveV2/shape_and_slicesConst*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-
ş
save_24/SaveV2SaveV2save_24/ShardedFilenamesave_24/SaveV2/tensor_namessave_24/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_24/control_dependencyIdentitysave_24/ShardedFilename^save_24/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_24/ShardedFilename
Ś
.save_24/MergeV2Checkpoints/checkpoint_prefixesPacksave_24/ShardedFilename^save_24/control_dependency*
N*

axis *
T0*
_output_shapes
:

save_24/MergeV2CheckpointsMergeV2Checkpoints.save_24/MergeV2Checkpoints/checkpoint_prefixessave_24/Const*
delete_old_dirs(

save_24/IdentityIdentitysave_24/Const^save_24/MergeV2Checkpoints^save_24/control_dependency*
_output_shapes
: *
T0
ö
save_24/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
Ă
"save_24/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ű
save_24/RestoreV2	RestoreV2save_24/Constsave_24/RestoreV2/tensor_names"save_24/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
¤
save_24/AssignAssignbeta1_powersave_24/RestoreV2* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(*
T0
¨
save_24/Assign_1Assignbeta2_powersave_24/RestoreV2:1*
T0*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(
Ż
save_24/Assign_2Assignpi/dense/biassave_24/RestoreV2:2*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes	
:*
use_locking(
ˇ
save_24/Assign_3Assignpi/dense/kernelsave_24/RestoreV2:3*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	<
ł
save_24/Assign_4Assignpi/dense_1/biassave_24/RestoreV2:4*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias
ź
save_24/Assign_5Assignpi/dense_1/kernelsave_24/RestoreV2:5*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:

˛
save_24/Assign_6Assignpi/dense_2/biassave_24/RestoreV2:6*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(
ť
save_24/Assign_7Assignpi/dense_2/kernelsave_24/RestoreV2:7*
_output_shapes
:	*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(
¨
save_24/Assign_8Assign
pi/log_stdsave_24/RestoreV2:8*
validate_shape(*
_class
loc:@pi/log_std*
T0*
use_locking(*
_output_shapes
:
Ż
save_24/Assign_9Assignvc/dense/biassave_24/RestoreV2:9*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
_output_shapes	
:
ś
save_24/Assign_10Assignvc/dense/bias/Adamsave_24/RestoreV2:10*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
use_locking(
¸
save_24/Assign_11Assignvc/dense/bias/Adam_1save_24/RestoreV2:11*
_output_shapes	
:*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(
š
save_24/Assign_12Assignvc/dense/kernelsave_24/RestoreV2:12*
_output_shapes
:	<*
T0*
validate_shape(*"
_class
loc:@vc/dense/kernel*
use_locking(
ž
save_24/Assign_13Assignvc/dense/kernel/Adamsave_24/RestoreV2:13*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	<
Ŕ
save_24/Assign_14Assignvc/dense/kernel/Adam_1save_24/RestoreV2:14*
validate_shape(*
_output_shapes
:	<*
T0*
use_locking(*"
_class
loc:@vc/dense/kernel
ľ
save_24/Assign_15Assignvc/dense_1/biassave_24/RestoreV2:15*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(
ş
save_24/Assign_16Assignvc/dense_1/bias/Adamsave_24/RestoreV2:16*
_output_shapes	
:*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(*
validate_shape(
ź
save_24/Assign_17Assignvc/dense_1/bias/Adam_1save_24/RestoreV2:17*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
ž
save_24/Assign_18Assignvc/dense_1/kernelsave_24/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(*
validate_shape(
Ă
save_24/Assign_19Assignvc/dense_1/kernel/Adamsave_24/RestoreV2:19*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:

Ĺ
save_24/Assign_20Assignvc/dense_1/kernel/Adam_1save_24/RestoreV2:20*
use_locking(*
validate_shape(* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel
´
save_24/Assign_21Assignvc/dense_2/biassave_24/RestoreV2:21*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias
š
save_24/Assign_22Assignvc/dense_2/bias/Adamsave_24/RestoreV2:22*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
ť
save_24/Assign_23Assignvc/dense_2/bias/Adam_1save_24/RestoreV2:23*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(
˝
save_24/Assign_24Assignvc/dense_2/kernelsave_24/RestoreV2:24*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	*
validate_shape(
Â
save_24/Assign_25Assignvc/dense_2/kernel/Adamsave_24/RestoreV2:25*
T0*
use_locking(*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
Ä
save_24/Assign_26Assignvc/dense_2/kernel/Adam_1save_24/RestoreV2:26*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
T0
ą
save_24/Assign_27Assignvf/dense/biassave_24/RestoreV2:27*
use_locking(* 
_class
loc:@vf/dense/bias*
T0*
validate_shape(*
_output_shapes	
:
ś
save_24/Assign_28Assignvf/dense/bias/Adamsave_24/RestoreV2:28*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
¸
save_24/Assign_29Assignvf/dense/bias/Adam_1save_24/RestoreV2:29*
validate_shape(* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:*
use_locking(
š
save_24/Assign_30Assignvf/dense/kernelsave_24/RestoreV2:30*
validate_shape(*
use_locking(*
_output_shapes
:	<*
T0*"
_class
loc:@vf/dense/kernel
ž
save_24/Assign_31Assignvf/dense/kernel/Adamsave_24/RestoreV2:31*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	<
Ŕ
save_24/Assign_32Assignvf/dense/kernel/Adam_1save_24/RestoreV2:32*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
validate_shape(*
use_locking(*
T0
ľ
save_24/Assign_33Assignvf/dense_1/biassave_24/RestoreV2:33*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(
ş
save_24/Assign_34Assignvf/dense_1/bias/Adamsave_24/RestoreV2:34*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
use_locking(
ź
save_24/Assign_35Assignvf/dense_1/bias/Adam_1save_24/RestoreV2:35*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
T0
ž
save_24/Assign_36Assignvf/dense_1/kernelsave_24/RestoreV2:36* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0
Ă
save_24/Assign_37Assignvf/dense_1/kernel/Adamsave_24/RestoreV2:37*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
T0*
use_locking(
Ĺ
save_24/Assign_38Assignvf/dense_1/kernel/Adam_1save_24/RestoreV2:38*
use_locking(*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
T0
´
save_24/Assign_39Assignvf/dense_2/biassave_24/RestoreV2:39*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
š
save_24/Assign_40Assignvf/dense_2/bias/Adamsave_24/RestoreV2:40*"
_class
loc:@vf/dense_2/bias*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
ť
save_24/Assign_41Assignvf/dense_2/bias/Adam_1save_24/RestoreV2:41*
use_locking(*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0
˝
save_24/Assign_42Assignvf/dense_2/kernelsave_24/RestoreV2:42*
_output_shapes
:	*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
T0
Â
save_24/Assign_43Assignvf/dense_2/kernel/Adamsave_24/RestoreV2:43*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
Ä
save_24/Assign_44Assignvf/dense_2/kernel/Adam_1save_24/RestoreV2:44*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel

save_24/restore_shardNoOp^save_24/Assign^save_24/Assign_1^save_24/Assign_10^save_24/Assign_11^save_24/Assign_12^save_24/Assign_13^save_24/Assign_14^save_24/Assign_15^save_24/Assign_16^save_24/Assign_17^save_24/Assign_18^save_24/Assign_19^save_24/Assign_2^save_24/Assign_20^save_24/Assign_21^save_24/Assign_22^save_24/Assign_23^save_24/Assign_24^save_24/Assign_25^save_24/Assign_26^save_24/Assign_27^save_24/Assign_28^save_24/Assign_29^save_24/Assign_3^save_24/Assign_30^save_24/Assign_31^save_24/Assign_32^save_24/Assign_33^save_24/Assign_34^save_24/Assign_35^save_24/Assign_36^save_24/Assign_37^save_24/Assign_38^save_24/Assign_39^save_24/Assign_4^save_24/Assign_40^save_24/Assign_41^save_24/Assign_42^save_24/Assign_43^save_24/Assign_44^save_24/Assign_5^save_24/Assign_6^save_24/Assign_7^save_24/Assign_8^save_24/Assign_9
3
save_24/restore_allNoOp^save_24/restore_shard
\
save_25/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_25/filenamePlaceholderWithDefaultsave_25/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_25/ConstPlaceholderWithDefaultsave_25/filename*
dtype0*
_output_shapes
: *
shape: 

save_25/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_09ad63e83f91469f8dafa904670270d3/part
~
save_25/StringJoin
StringJoinsave_25/Constsave_25/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_25/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_25/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_25/ShardedFilenameShardedFilenamesave_25/StringJoinsave_25/ShardedFilename/shardsave_25/num_shards*
_output_shapes
: 
ó
save_25/SaveV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
Ŕ
save_25/SaveV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ş
save_25/SaveV2SaveV2save_25/ShardedFilenamesave_25/SaveV2/tensor_namessave_25/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_25/control_dependencyIdentitysave_25/ShardedFilename^save_25/SaveV2**
_class 
loc:@save_25/ShardedFilename*
_output_shapes
: *
T0
Ś
.save_25/MergeV2Checkpoints/checkpoint_prefixesPacksave_25/ShardedFilename^save_25/control_dependency*
N*
_output_shapes
:*

axis *
T0

save_25/MergeV2CheckpointsMergeV2Checkpoints.save_25/MergeV2Checkpoints/checkpoint_prefixessave_25/Const*
delete_old_dirs(

save_25/IdentityIdentitysave_25/Const^save_25/MergeV2Checkpoints^save_25/control_dependency*
T0*
_output_shapes
: 
ö
save_25/RestoreV2/tensor_namesConst*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-
Ă
"save_25/RestoreV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
ű
save_25/RestoreV2	RestoreV2save_25/Constsave_25/RestoreV2/tensor_names"save_25/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
¤
save_25/AssignAssignbeta1_powersave_25/RestoreV2* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: *
validate_shape(*
use_locking(
¨
save_25/Assign_1Assignbeta2_powersave_25/RestoreV2:1*
_output_shapes
: *
validate_shape(*
T0*
use_locking(* 
_class
loc:@vc/dense/bias
Ż
save_25/Assign_2Assignpi/dense/biassave_25/RestoreV2:2* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ˇ
save_25/Assign_3Assignpi/dense/kernelsave_25/RestoreV2:3*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	<
ł
save_25/Assign_4Assignpi/dense_1/biassave_25/RestoreV2:4*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
use_locking(
ź
save_25/Assign_5Assignpi/dense_1/kernelsave_25/RestoreV2:5*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
T0
˛
save_25/Assign_6Assignpi/dense_2/biassave_25/RestoreV2:6*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
ť
save_25/Assign_7Assignpi/dense_2/kernelsave_25/RestoreV2:7*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
¨
save_25/Assign_8Assign
pi/log_stdsave_25/RestoreV2:8*
_class
loc:@pi/log_std*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
Ż
save_25/Assign_9Assignvc/dense/biassave_25/RestoreV2:9* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
ś
save_25/Assign_10Assignvc/dense/bias/Adamsave_25/RestoreV2:10*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
use_locking(
¸
save_25/Assign_11Assignvc/dense/bias/Adam_1save_25/RestoreV2:11*
T0*
use_locking(*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:
š
save_25/Assign_12Assignvc/dense/kernelsave_25/RestoreV2:12*
_output_shapes
:	<*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel
ž
save_25/Assign_13Assignvc/dense/kernel/Adamsave_25/RestoreV2:13*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<
Ŕ
save_25/Assign_14Assignvc/dense/kernel/Adam_1save_25/RestoreV2:14*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
use_locking(
ľ
save_25/Assign_15Assignvc/dense_1/biassave_25/RestoreV2:15*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:
ş
save_25/Assign_16Assignvc/dense_1/bias/Adamsave_25/RestoreV2:16*
T0*
_output_shapes	
:*
use_locking(*"
_class
loc:@vc/dense_1/bias*
validate_shape(
ź
save_25/Assign_17Assignvc/dense_1/bias/Adam_1save_25/RestoreV2:17*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(*"
_class
loc:@vc/dense_1/bias
ž
save_25/Assign_18Assignvc/dense_1/kernelsave_25/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(
Ă
save_25/Assign_19Assignvc/dense_1/kernel/Adamsave_25/RestoreV2:19*
T0* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
Ĺ
save_25/Assign_20Assignvc/dense_1/kernel/Adam_1save_25/RestoreV2:20*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0
´
save_25/Assign_21Assignvc/dense_2/biassave_25/RestoreV2:21*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
T0*
use_locking(
š
save_25/Assign_22Assignvc/dense_2/bias/Adamsave_25/RestoreV2:22*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0
ť
save_25/Assign_23Assignvc/dense_2/bias/Adam_1save_25/RestoreV2:23*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
˝
save_25/Assign_24Assignvc/dense_2/kernelsave_25/RestoreV2:24*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel
Â
save_25/Assign_25Assignvc/dense_2/kernel/Adamsave_25/RestoreV2:25*
T0*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	
Ä
save_25/Assign_26Assignvc/dense_2/kernel/Adam_1save_25/RestoreV2:26*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	
ą
save_25/Assign_27Assignvf/dense/biassave_25/RestoreV2:27*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
ś
save_25/Assign_28Assignvf/dense/bias/Adamsave_25/RestoreV2:28*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vf/dense/bias
¸
save_25/Assign_29Assignvf/dense/bias/Adam_1save_25/RestoreV2:29*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0*
use_locking(
š
save_25/Assign_30Assignvf/dense/kernelsave_25/RestoreV2:30*
validate_shape(*
_output_shapes
:	<*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel
ž
save_25/Assign_31Assignvf/dense/kernel/Adamsave_25/RestoreV2:31*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(*
T0
Ŕ
save_25/Assign_32Assignvf/dense/kernel/Adam_1save_25/RestoreV2:32*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(*
use_locking(
ľ
save_25/Assign_33Assignvf/dense_1/biassave_25/RestoreV2:33*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias
ş
save_25/Assign_34Assignvf/dense_1/bias/Adamsave_25/RestoreV2:34*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:
ź
save_25/Assign_35Assignvf/dense_1/bias/Adam_1save_25/RestoreV2:35*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(*
T0
ž
save_25/Assign_36Assignvf/dense_1/kernelsave_25/RestoreV2:36*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:

Ă
save_25/Assign_37Assignvf/dense_1/kernel/Adamsave_25/RestoreV2:37* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0*
validate_shape(
Ĺ
save_25/Assign_38Assignvf/dense_1/kernel/Adam_1save_25/RestoreV2:38*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:

´
save_25/Assign_39Assignvf/dense_2/biassave_25/RestoreV2:39*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
š
save_25/Assign_40Assignvf/dense_2/bias/Adamsave_25/RestoreV2:40*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
use_locking(*
validate_shape(
ť
save_25/Assign_41Assignvf/dense_2/bias/Adam_1save_25/RestoreV2:41*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0*
use_locking(
˝
save_25/Assign_42Assignvf/dense_2/kernelsave_25/RestoreV2:42*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0*$
_class
loc:@vf/dense_2/kernel
Â
save_25/Assign_43Assignvf/dense_2/kernel/Adamsave_25/RestoreV2:43*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ä
save_25/Assign_44Assignvf/dense_2/kernel/Adam_1save_25/RestoreV2:44*
validate_shape(*
_output_shapes
:	*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0

save_25/restore_shardNoOp^save_25/Assign^save_25/Assign_1^save_25/Assign_10^save_25/Assign_11^save_25/Assign_12^save_25/Assign_13^save_25/Assign_14^save_25/Assign_15^save_25/Assign_16^save_25/Assign_17^save_25/Assign_18^save_25/Assign_19^save_25/Assign_2^save_25/Assign_20^save_25/Assign_21^save_25/Assign_22^save_25/Assign_23^save_25/Assign_24^save_25/Assign_25^save_25/Assign_26^save_25/Assign_27^save_25/Assign_28^save_25/Assign_29^save_25/Assign_3^save_25/Assign_30^save_25/Assign_31^save_25/Assign_32^save_25/Assign_33^save_25/Assign_34^save_25/Assign_35^save_25/Assign_36^save_25/Assign_37^save_25/Assign_38^save_25/Assign_39^save_25/Assign_4^save_25/Assign_40^save_25/Assign_41^save_25/Assign_42^save_25/Assign_43^save_25/Assign_44^save_25/Assign_5^save_25/Assign_6^save_25/Assign_7^save_25/Assign_8^save_25/Assign_9
3
save_25/restore_allNoOp^save_25/restore_shard
\
save_26/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_26/filenamePlaceholderWithDefaultsave_26/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_26/ConstPlaceholderWithDefaultsave_26/filename*
_output_shapes
: *
dtype0*
shape: 

save_26/StringJoin/inputs_1Const*<
value3B1 B+_temp_4d78bf1410904c1799c7c084c4258387/part*
dtype0*
_output_shapes
: 
~
save_26/StringJoin
StringJoinsave_26/Constsave_26/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_26/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_26/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_26/ShardedFilenameShardedFilenamesave_26/StringJoinsave_26/ShardedFilename/shardsave_26/num_shards*
_output_shapes
: 
ó
save_26/SaveV2/tensor_namesConst*
_output_shapes
:-*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ŕ
save_26/SaveV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
ş
save_26/SaveV2SaveV2save_26/ShardedFilenamesave_26/SaveV2/tensor_namessave_26/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_26/control_dependencyIdentitysave_26/ShardedFilename^save_26/SaveV2*
T0**
_class 
loc:@save_26/ShardedFilename*
_output_shapes
: 
Ś
.save_26/MergeV2Checkpoints/checkpoint_prefixesPacksave_26/ShardedFilename^save_26/control_dependency*
T0*

axis *
_output_shapes
:*
N

save_26/MergeV2CheckpointsMergeV2Checkpoints.save_26/MergeV2Checkpoints/checkpoint_prefixessave_26/Const*
delete_old_dirs(

save_26/IdentityIdentitysave_26/Const^save_26/MergeV2Checkpoints^save_26/control_dependency*
T0*
_output_shapes
: 
ö
save_26/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
Ă
"save_26/RestoreV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
ű
save_26/RestoreV2	RestoreV2save_26/Constsave_26/RestoreV2/tensor_names"save_26/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
¤
save_26/AssignAssignbeta1_powersave_26/RestoreV2*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(
¨
save_26/Assign_1Assignbeta2_powersave_26/RestoreV2:1*
T0*
validate_shape(*
_output_shapes
: *
use_locking(* 
_class
loc:@vc/dense/bias
Ż
save_26/Assign_2Assignpi/dense/biassave_26/RestoreV2:2*
_output_shapes	
:*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(
ˇ
save_26/Assign_3Assignpi/dense/kernelsave_26/RestoreV2:3*
T0*
_output_shapes
:	<*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel
ł
save_26/Assign_4Assignpi/dense_1/biassave_26/RestoreV2:4*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(
ź
save_26/Assign_5Assignpi/dense_1/kernelsave_26/RestoreV2:5*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:
*
validate_shape(
˛
save_26/Assign_6Assignpi/dense_2/biassave_26/RestoreV2:6*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(
ť
save_26/Assign_7Assignpi/dense_2/kernelsave_26/RestoreV2:7*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
¨
save_26/Assign_8Assign
pi/log_stdsave_26/RestoreV2:8*
T0*
_output_shapes
:*
use_locking(*
_class
loc:@pi/log_std*
validate_shape(
Ż
save_26/Assign_9Assignvc/dense/biassave_26/RestoreV2:9*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0
ś
save_26/Assign_10Assignvc/dense/bias/Adamsave_26/RestoreV2:10*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(
¸
save_26/Assign_11Assignvc/dense/bias/Adam_1save_26/RestoreV2:11*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias
š
save_26/Assign_12Assignvc/dense/kernelsave_26/RestoreV2:12*
use_locking(*
_output_shapes
:	<*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(
ž
save_26/Assign_13Assignvc/dense/kernel/Adamsave_26/RestoreV2:13*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	<*
use_locking(
Ŕ
save_26/Assign_14Assignvc/dense/kernel/Adam_1save_26/RestoreV2:14*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel
ľ
save_26/Assign_15Assignvc/dense_1/biassave_26/RestoreV2:15*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
validate_shape(
ş
save_26/Assign_16Assignvc/dense_1/bias/Adamsave_26/RestoreV2:16*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(*
T0
ź
save_26/Assign_17Assignvc/dense_1/bias/Adam_1save_26/RestoreV2:17*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(
ž
save_26/Assign_18Assignvc/dense_1/kernelsave_26/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:

Ă
save_26/Assign_19Assignvc/dense_1/kernel/Adamsave_26/RestoreV2:19*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

Ĺ
save_26/Assign_20Assignvc/dense_1/kernel/Adam_1save_26/RestoreV2:20*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

´
save_26/Assign_21Assignvc/dense_2/biassave_26/RestoreV2:21*"
_class
loc:@vc/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
š
save_26/Assign_22Assignvc/dense_2/bias/Adamsave_26/RestoreV2:22*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias
ť
save_26/Assign_23Assignvc/dense_2/bias/Adam_1save_26/RestoreV2:23*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
˝
save_26/Assign_24Assignvc/dense_2/kernelsave_26/RestoreV2:24*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
Â
save_26/Assign_25Assignvc/dense_2/kernel/Adamsave_26/RestoreV2:25*
validate_shape(*
use_locking(*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel
Ä
save_26/Assign_26Assignvc/dense_2/kernel/Adam_1save_26/RestoreV2:26*
validate_shape(*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	
ą
save_26/Assign_27Assignvf/dense/biassave_26/RestoreV2:27* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0
ś
save_26/Assign_28Assignvf/dense/bias/Adamsave_26/RestoreV2:28*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
T0*
validate_shape(
¸
save_26/Assign_29Assignvf/dense/bias/Adam_1save_26/RestoreV2:29*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
š
save_26/Assign_30Assignvf/dense/kernelsave_26/RestoreV2:30*"
_class
loc:@vf/dense/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	<
ž
save_26/Assign_31Assignvf/dense/kernel/Adamsave_26/RestoreV2:31*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	<
Ŕ
save_26/Assign_32Assignvf/dense/kernel/Adam_1save_26/RestoreV2:32*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	<*
T0
ľ
save_26/Assign_33Assignvf/dense_1/biassave_26/RestoreV2:33*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:*
T0
ş
save_26/Assign_34Assignvf/dense_1/bias/Adamsave_26/RestoreV2:34*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias
ź
save_26/Assign_35Assignvf/dense_1/bias/Adam_1save_26/RestoreV2:35*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(*"
_class
loc:@vf/dense_1/bias
ž
save_26/Assign_36Assignvf/dense_1/kernelsave_26/RestoreV2:36*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
Ă
save_26/Assign_37Assignvf/dense_1/kernel/Adamsave_26/RestoreV2:37*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(
Ĺ
save_26/Assign_38Assignvf/dense_1/kernel/Adam_1save_26/RestoreV2:38*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
´
save_26/Assign_39Assignvf/dense_2/biassave_26/RestoreV2:39*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(
š
save_26/Assign_40Assignvf/dense_2/bias/Adamsave_26/RestoreV2:40*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
T0
ť
save_26/Assign_41Assignvf/dense_2/bias/Adam_1save_26/RestoreV2:41*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias
˝
save_26/Assign_42Assignvf/dense_2/kernelsave_26/RestoreV2:42*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
use_locking(*
validate_shape(
Â
save_26/Assign_43Assignvf/dense_2/kernel/Adamsave_26/RestoreV2:43*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
Ä
save_26/Assign_44Assignvf/dense_2/kernel/Adam_1save_26/RestoreV2:44*$
_class
loc:@vf/dense_2/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	

save_26/restore_shardNoOp^save_26/Assign^save_26/Assign_1^save_26/Assign_10^save_26/Assign_11^save_26/Assign_12^save_26/Assign_13^save_26/Assign_14^save_26/Assign_15^save_26/Assign_16^save_26/Assign_17^save_26/Assign_18^save_26/Assign_19^save_26/Assign_2^save_26/Assign_20^save_26/Assign_21^save_26/Assign_22^save_26/Assign_23^save_26/Assign_24^save_26/Assign_25^save_26/Assign_26^save_26/Assign_27^save_26/Assign_28^save_26/Assign_29^save_26/Assign_3^save_26/Assign_30^save_26/Assign_31^save_26/Assign_32^save_26/Assign_33^save_26/Assign_34^save_26/Assign_35^save_26/Assign_36^save_26/Assign_37^save_26/Assign_38^save_26/Assign_39^save_26/Assign_4^save_26/Assign_40^save_26/Assign_41^save_26/Assign_42^save_26/Assign_43^save_26/Assign_44^save_26/Assign_5^save_26/Assign_6^save_26/Assign_7^save_26/Assign_8^save_26/Assign_9
3
save_26/restore_allNoOp^save_26/restore_shard
\
save_27/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_27/filenamePlaceholderWithDefaultsave_27/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_27/ConstPlaceholderWithDefaultsave_27/filename*
dtype0*
shape: *
_output_shapes
: 

save_27/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_e01c6744b03e4f73b36daf722252f789/part
~
save_27/StringJoin
StringJoinsave_27/Constsave_27/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_27/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_27/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_27/ShardedFilenameShardedFilenamesave_27/StringJoinsave_27/ShardedFilename/shardsave_27/num_shards*
_output_shapes
: 
ó
save_27/SaveV2/tensor_namesConst*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-
Ŕ
save_27/SaveV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ş
save_27/SaveV2SaveV2save_27/ShardedFilenamesave_27/SaveV2/tensor_namessave_27/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_27/control_dependencyIdentitysave_27/ShardedFilename^save_27/SaveV2*
T0**
_class 
loc:@save_27/ShardedFilename*
_output_shapes
: 
Ś
.save_27/MergeV2Checkpoints/checkpoint_prefixesPacksave_27/ShardedFilename^save_27/control_dependency*
T0*
N*
_output_shapes
:*

axis 

save_27/MergeV2CheckpointsMergeV2Checkpoints.save_27/MergeV2Checkpoints/checkpoint_prefixessave_27/Const*
delete_old_dirs(

save_27/IdentityIdentitysave_27/Const^save_27/MergeV2Checkpoints^save_27/control_dependency*
T0*
_output_shapes
: 
ö
save_27/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
Ă
"save_27/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ű
save_27/RestoreV2	RestoreV2save_27/Constsave_27/RestoreV2/tensor_names"save_27/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
¤
save_27/AssignAssignbeta1_powersave_27/RestoreV2*
_output_shapes
: *
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
use_locking(
¨
save_27/Assign_1Assignbeta2_powersave_27/RestoreV2:1*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
T0
Ż
save_27/Assign_2Assignpi/dense/biassave_27/RestoreV2:2*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0* 
_class
loc:@pi/dense/bias
ˇ
save_27/Assign_3Assignpi/dense/kernelsave_27/RestoreV2:3*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	<*
T0
ł
save_27/Assign_4Assignpi/dense_1/biassave_27/RestoreV2:4*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(*"
_class
loc:@pi/dense_1/bias
ź
save_27/Assign_5Assignpi/dense_1/kernelsave_27/RestoreV2:5*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

˛
save_27/Assign_6Assignpi/dense_2/biassave_27/RestoreV2:6*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
ť
save_27/Assign_7Assignpi/dense_2/kernelsave_27/RestoreV2:7*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel
¨
save_27/Assign_8Assign
pi/log_stdsave_27/RestoreV2:8*
use_locking(*
_class
loc:@pi/log_std*
T0*
_output_shapes
:*
validate_shape(
Ż
save_27/Assign_9Assignvc/dense/biassave_27/RestoreV2:9*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
use_locking(
ś
save_27/Assign_10Assignvc/dense/bias/Adamsave_27/RestoreV2:10*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
¸
save_27/Assign_11Assignvc/dense/bias/Adam_1save_27/RestoreV2:11*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vc/dense/bias
š
save_27/Assign_12Assignvc/dense/kernelsave_27/RestoreV2:12*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<
ž
save_27/Assign_13Assignvc/dense/kernel/Adamsave_27/RestoreV2:13*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<*
validate_shape(
Ŕ
save_27/Assign_14Assignvc/dense/kernel/Adam_1save_27/RestoreV2:14*
T0*
use_locking(*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	<
ľ
save_27/Assign_15Assignvc/dense_1/biassave_27/RestoreV2:15*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias
ş
save_27/Assign_16Assignvc/dense_1/bias/Adamsave_27/RestoreV2:16*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(
ź
save_27/Assign_17Assignvc/dense_1/bias/Adam_1save_27/RestoreV2:17*
_output_shapes	
:*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(
ž
save_27/Assign_18Assignvc/dense_1/kernelsave_27/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:
*
use_locking(
Ă
save_27/Assign_19Assignvc/dense_1/kernel/Adamsave_27/RestoreV2:19*
T0*
use_locking(* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
Ĺ
save_27/Assign_20Assignvc/dense_1/kernel/Adam_1save_27/RestoreV2:20* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(
´
save_27/Assign_21Assignvc/dense_2/biassave_27/RestoreV2:21*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:*
use_locking(
š
save_27/Assign_22Assignvc/dense_2/bias/Adamsave_27/RestoreV2:22*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(
ť
save_27/Assign_23Assignvc/dense_2/bias/Adam_1save_27/RestoreV2:23*
_output_shapes
:*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias*
use_locking(
˝
save_27/Assign_24Assignvc/dense_2/kernelsave_27/RestoreV2:24*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
T0
Â
save_27/Assign_25Assignvc/dense_2/kernel/Adamsave_27/RestoreV2:25*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	
Ä
save_27/Assign_26Assignvc/dense_2/kernel/Adam_1save_27/RestoreV2:26*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes
:	
ą
save_27/Assign_27Assignvf/dense/biassave_27/RestoreV2:27*
_output_shapes	
:*
use_locking(* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0
ś
save_27/Assign_28Assignvf/dense/bias/Adamsave_27/RestoreV2:28*
use_locking(* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:*
T0
¸
save_27/Assign_29Assignvf/dense/bias/Adam_1save_27/RestoreV2:29* 
_class
loc:@vf/dense/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
š
save_27/Assign_30Assignvf/dense/kernelsave_27/RestoreV2:30*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	<
ž
save_27/Assign_31Assignvf/dense/kernel/Adamsave_27/RestoreV2:31*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(
Ŕ
save_27/Assign_32Assignvf/dense/kernel/Adam_1save_27/RestoreV2:32*"
_class
loc:@vf/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	<*
use_locking(
ľ
save_27/Assign_33Assignvf/dense_1/biassave_27/RestoreV2:33*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(*
T0
ş
save_27/Assign_34Assignvf/dense_1/bias/Adamsave_27/RestoreV2:34*"
_class
loc:@vf/dense_1/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:
ź
save_27/Assign_35Assignvf/dense_1/bias/Adam_1save_27/RestoreV2:35*
use_locking(*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0
ž
save_27/Assign_36Assignvf/dense_1/kernelsave_27/RestoreV2:36*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:

Ă
save_27/Assign_37Assignvf/dense_1/kernel/Adamsave_27/RestoreV2:37*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:
*
validate_shape(*
use_locking(
Ĺ
save_27/Assign_38Assignvf/dense_1/kernel/Adam_1save_27/RestoreV2:38*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:

´
save_27/Assign_39Assignvf/dense_2/biassave_27/RestoreV2:39*"
_class
loc:@vf/dense_2/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:
š
save_27/Assign_40Assignvf/dense_2/bias/Adamsave_27/RestoreV2:40*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:
ť
save_27/Assign_41Assignvf/dense_2/bias/Adam_1save_27/RestoreV2:41*
use_locking(*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0
˝
save_27/Assign_42Assignvf/dense_2/kernelsave_27/RestoreV2:42*
validate_shape(*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
Â
save_27/Assign_43Assignvf/dense_2/kernel/Adamsave_27/RestoreV2:43*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel
Ä
save_27/Assign_44Assignvf/dense_2/kernel/Adam_1save_27/RestoreV2:44*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	

save_27/restore_shardNoOp^save_27/Assign^save_27/Assign_1^save_27/Assign_10^save_27/Assign_11^save_27/Assign_12^save_27/Assign_13^save_27/Assign_14^save_27/Assign_15^save_27/Assign_16^save_27/Assign_17^save_27/Assign_18^save_27/Assign_19^save_27/Assign_2^save_27/Assign_20^save_27/Assign_21^save_27/Assign_22^save_27/Assign_23^save_27/Assign_24^save_27/Assign_25^save_27/Assign_26^save_27/Assign_27^save_27/Assign_28^save_27/Assign_29^save_27/Assign_3^save_27/Assign_30^save_27/Assign_31^save_27/Assign_32^save_27/Assign_33^save_27/Assign_34^save_27/Assign_35^save_27/Assign_36^save_27/Assign_37^save_27/Assign_38^save_27/Assign_39^save_27/Assign_4^save_27/Assign_40^save_27/Assign_41^save_27/Assign_42^save_27/Assign_43^save_27/Assign_44^save_27/Assign_5^save_27/Assign_6^save_27/Assign_7^save_27/Assign_8^save_27/Assign_9
3
save_27/restore_allNoOp^save_27/restore_shard
\
save_28/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_28/filenamePlaceholderWithDefaultsave_28/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_28/ConstPlaceholderWithDefaultsave_28/filename*
_output_shapes
: *
dtype0*
shape: 

save_28/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_7d180d354aca472b9583f05ba6e54852/part
~
save_28/StringJoin
StringJoinsave_28/Constsave_28/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
T
save_28/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_28/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_28/ShardedFilenameShardedFilenamesave_28/StringJoinsave_28/ShardedFilename/shardsave_28/num_shards*
_output_shapes
: 
ó
save_28/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ŕ
save_28/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ş
save_28/SaveV2SaveV2save_28/ShardedFilenamesave_28/SaveV2/tensor_namessave_28/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_28/control_dependencyIdentitysave_28/ShardedFilename^save_28/SaveV2*
_output_shapes
: **
_class 
loc:@save_28/ShardedFilename*
T0
Ś
.save_28/MergeV2Checkpoints/checkpoint_prefixesPacksave_28/ShardedFilename^save_28/control_dependency*

axis *
_output_shapes
:*
T0*
N

save_28/MergeV2CheckpointsMergeV2Checkpoints.save_28/MergeV2Checkpoints/checkpoint_prefixessave_28/Const*
delete_old_dirs(

save_28/IdentityIdentitysave_28/Const^save_28/MergeV2Checkpoints^save_28/control_dependency*
T0*
_output_shapes
: 
ö
save_28/RestoreV2/tensor_namesConst*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
Ă
"save_28/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ű
save_28/RestoreV2	RestoreV2save_28/Constsave_28/RestoreV2/tensor_names"save_28/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
¤
save_28/AssignAssignbeta1_powersave_28/RestoreV2* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes
: *
T0*
validate_shape(
¨
save_28/Assign_1Assignbeta2_powersave_28/RestoreV2:1* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: *
validate_shape(*
use_locking(
Ż
save_28/Assign_2Assignpi/dense/biassave_28/RestoreV2:2*
validate_shape(*
_output_shapes	
:*
use_locking(* 
_class
loc:@pi/dense/bias*
T0
ˇ
save_28/Assign_3Assignpi/dense/kernelsave_28/RestoreV2:3*
_output_shapes
:	<*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel
ł
save_28/Assign_4Assignpi/dense_1/biassave_28/RestoreV2:4*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(*"
_class
loc:@pi/dense_1/bias
ź
save_28/Assign_5Assignpi/dense_1/kernelsave_28/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
˛
save_28/Assign_6Assignpi/dense_2/biassave_28/RestoreV2:6*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
ť
save_28/Assign_7Assignpi/dense_2/kernelsave_28/RestoreV2:7*
validate_shape(*
use_locking(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0
¨
save_28/Assign_8Assign
pi/log_stdsave_28/RestoreV2:8*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Ż
save_28/Assign_9Assignvc/dense/biassave_28/RestoreV2:9*
_output_shapes	
:*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(
ś
save_28/Assign_10Assignvc/dense/bias/Adamsave_28/RestoreV2:10* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:
¸
save_28/Assign_11Assignvc/dense/bias/Adam_1save_28/RestoreV2:11*
use_locking(*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0
š
save_28/Assign_12Assignvc/dense/kernelsave_28/RestoreV2:12*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	<
ž
save_28/Assign_13Assignvc/dense/kernel/Adamsave_28/RestoreV2:13*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0
Ŕ
save_28/Assign_14Assignvc/dense/kernel/Adam_1save_28/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	<*
T0
ľ
save_28/Assign_15Assignvc/dense_1/biassave_28/RestoreV2:15*
use_locking(*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(
ş
save_28/Assign_16Assignvc/dense_1/bias/Adamsave_28/RestoreV2:16*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias
ź
save_28/Assign_17Assignvc/dense_1/bias/Adam_1save_28/RestoreV2:17*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(
ž
save_28/Assign_18Assignvc/dense_1/kernelsave_28/RestoreV2:18*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
Ă
save_28/Assign_19Assignvc/dense_1/kernel/Adamsave_28/RestoreV2:19*
T0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(
Ĺ
save_28/Assign_20Assignvc/dense_1/kernel/Adam_1save_28/RestoreV2:20*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking(
´
save_28/Assign_21Assignvc/dense_2/biassave_28/RestoreV2:21*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
š
save_28/Assign_22Assignvc/dense_2/bias/Adamsave_28/RestoreV2:22*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias
ť
save_28/Assign_23Assignvc/dense_2/bias/Adam_1save_28/RestoreV2:23*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(*
validate_shape(
˝
save_28/Assign_24Assignvc/dense_2/kernelsave_28/RestoreV2:24*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	*
T0
Â
save_28/Assign_25Assignvc/dense_2/kernel/Adamsave_28/RestoreV2:25*
validate_shape(*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
T0*
use_locking(
Ä
save_28/Assign_26Assignvc/dense_2/kernel/Adam_1save_28/RestoreV2:26*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
T0*
use_locking(
ą
save_28/Assign_27Assignvf/dense/biassave_28/RestoreV2:27*
use_locking(* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:*
validate_shape(
ś
save_28/Assign_28Assignvf/dense/bias/Adamsave_28/RestoreV2:28*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias
¸
save_28/Assign_29Assignvf/dense/bias/Adam_1save_28/RestoreV2:29* 
_class
loc:@vf/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:
š
save_28/Assign_30Assignvf/dense/kernelsave_28/RestoreV2:30*
validate_shape(*"
_class
loc:@vf/dense/kernel*
use_locking(*
T0*
_output_shapes
:	<
ž
save_28/Assign_31Assignvf/dense/kernel/Adamsave_28/RestoreV2:31*
validate_shape(*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vf/dense/kernel*
T0
Ŕ
save_28/Assign_32Assignvf/dense/kernel/Adam_1save_28/RestoreV2:32*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	<*
T0*
use_locking(
ľ
save_28/Assign_33Assignvf/dense_1/biassave_28/RestoreV2:33*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(*
T0
ş
save_28/Assign_34Assignvf/dense_1/bias/Adamsave_28/RestoreV2:34*
_output_shapes	
:*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0
ź
save_28/Assign_35Assignvf/dense_1/bias/Adam_1save_28/RestoreV2:35*
validate_shape(*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(
ž
save_28/Assign_36Assignvf/dense_1/kernelsave_28/RestoreV2:36* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(
Ă
save_28/Assign_37Assignvf/dense_1/kernel/Adamsave_28/RestoreV2:37* 
_output_shapes
:
*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel
Ĺ
save_28/Assign_38Assignvf/dense_1/kernel/Adam_1save_28/RestoreV2:38*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
validate_shape(*
T0
´
save_28/Assign_39Assignvf/dense_2/biassave_28/RestoreV2:39*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias*
validate_shape(
š
save_28/Assign_40Assignvf/dense_2/bias/Adamsave_28/RestoreV2:40*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
ť
save_28/Assign_41Assignvf/dense_2/bias/Adam_1save_28/RestoreV2:41*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
use_locking(
˝
save_28/Assign_42Assignvf/dense_2/kernelsave_28/RestoreV2:42*
validate_shape(*
T0*
_output_shapes
:	*
use_locking(*$
_class
loc:@vf/dense_2/kernel
Â
save_28/Assign_43Assignvf/dense_2/kernel/Adamsave_28/RestoreV2:43*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	
Ä
save_28/Assign_44Assignvf/dense_2/kernel/Adam_1save_28/RestoreV2:44*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	

save_28/restore_shardNoOp^save_28/Assign^save_28/Assign_1^save_28/Assign_10^save_28/Assign_11^save_28/Assign_12^save_28/Assign_13^save_28/Assign_14^save_28/Assign_15^save_28/Assign_16^save_28/Assign_17^save_28/Assign_18^save_28/Assign_19^save_28/Assign_2^save_28/Assign_20^save_28/Assign_21^save_28/Assign_22^save_28/Assign_23^save_28/Assign_24^save_28/Assign_25^save_28/Assign_26^save_28/Assign_27^save_28/Assign_28^save_28/Assign_29^save_28/Assign_3^save_28/Assign_30^save_28/Assign_31^save_28/Assign_32^save_28/Assign_33^save_28/Assign_34^save_28/Assign_35^save_28/Assign_36^save_28/Assign_37^save_28/Assign_38^save_28/Assign_39^save_28/Assign_4^save_28/Assign_40^save_28/Assign_41^save_28/Assign_42^save_28/Assign_43^save_28/Assign_44^save_28/Assign_5^save_28/Assign_6^save_28/Assign_7^save_28/Assign_8^save_28/Assign_9
3
save_28/restore_allNoOp^save_28/restore_shard
\
save_29/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
t
save_29/filenamePlaceholderWithDefaultsave_29/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_29/ConstPlaceholderWithDefaultsave_29/filename*
dtype0*
_output_shapes
: *
shape: 

save_29/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_de7aaa34782848c886479c3fa0f1af7e/part*
dtype0
~
save_29/StringJoin
StringJoinsave_29/Constsave_29/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_29/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_29/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_29/ShardedFilenameShardedFilenamesave_29/StringJoinsave_29/ShardedFilename/shardsave_29/num_shards*
_output_shapes
: 
ó
save_29/SaveV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
Ŕ
save_29/SaveV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ş
save_29/SaveV2SaveV2save_29/ShardedFilenamesave_29/SaveV2/tensor_namessave_29/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_29/control_dependencyIdentitysave_29/ShardedFilename^save_29/SaveV2**
_class 
loc:@save_29/ShardedFilename*
T0*
_output_shapes
: 
Ś
.save_29/MergeV2Checkpoints/checkpoint_prefixesPacksave_29/ShardedFilename^save_29/control_dependency*
N*
_output_shapes
:*
T0*

axis 

save_29/MergeV2CheckpointsMergeV2Checkpoints.save_29/MergeV2Checkpoints/checkpoint_prefixessave_29/Const*
delete_old_dirs(

save_29/IdentityIdentitysave_29/Const^save_29/MergeV2Checkpoints^save_29/control_dependency*
T0*
_output_shapes
: 
ö
save_29/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
Ă
"save_29/RestoreV2/shape_and_slicesConst*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-
ű
save_29/RestoreV2	RestoreV2save_29/Constsave_29/RestoreV2/tensor_names"save_29/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
¤
save_29/AssignAssignbeta1_powersave_29/RestoreV2*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
T0
¨
save_29/Assign_1Assignbeta2_powersave_29/RestoreV2:1* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
: 
Ż
save_29/Assign_2Assignpi/dense/biassave_29/RestoreV2:2*
use_locking(*
T0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias*
validate_shape(
ˇ
save_29/Assign_3Assignpi/dense/kernelsave_29/RestoreV2:3*
use_locking(*
T0*
_output_shapes
:	<*"
_class
loc:@pi/dense/kernel*
validate_shape(
ł
save_29/Assign_4Assignpi/dense_1/biassave_29/RestoreV2:4*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:*
use_locking(*
T0
ź
save_29/Assign_5Assignpi/dense_1/kernelsave_29/RestoreV2:5*
use_locking(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0
˛
save_29/Assign_6Assignpi/dense_2/biassave_29/RestoreV2:6*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0
ť
save_29/Assign_7Assignpi/dense_2/kernelsave_29/RestoreV2:7*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	*
validate_shape(
¨
save_29/Assign_8Assign
pi/log_stdsave_29/RestoreV2:8*
_class
loc:@pi/log_std*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Ż
save_29/Assign_9Assignvc/dense/biassave_29/RestoreV2:9*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ś
save_29/Assign_10Assignvc/dense/bias/Adamsave_29/RestoreV2:10*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
use_locking(
¸
save_29/Assign_11Assignvc/dense/bias/Adam_1save_29/RestoreV2:11*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes	
:
š
save_29/Assign_12Assignvc/dense/kernelsave_29/RestoreV2:12*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<*
use_locking(*
validate_shape(
ž
save_29/Assign_13Assignvc/dense/kernel/Adamsave_29/RestoreV2:13*
validate_shape(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
use_locking(*
T0
Ŕ
save_29/Assign_14Assignvc/dense/kernel/Adam_1save_29/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	<*
use_locking(
ľ
save_29/Assign_15Assignvc/dense_1/biassave_29/RestoreV2:15*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
ş
save_29/Assign_16Assignvc/dense_1/bias/Adamsave_29/RestoreV2:16*
_output_shapes	
:*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
validate_shape(
ź
save_29/Assign_17Assignvc/dense_1/bias/Adam_1save_29/RestoreV2:17*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
ž
save_29/Assign_18Assignvc/dense_1/kernelsave_29/RestoreV2:18* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0
Ă
save_29/Assign_19Assignvc/dense_1/kernel/Adamsave_29/RestoreV2:19*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0
Ĺ
save_29/Assign_20Assignvc/dense_1/kernel/Adam_1save_29/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
´
save_29/Assign_21Assignvc/dense_2/biassave_29/RestoreV2:21*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0
š
save_29/Assign_22Assignvc/dense_2/bias/Adamsave_29/RestoreV2:22*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
ť
save_29/Assign_23Assignvc/dense_2/bias/Adam_1save_29/RestoreV2:23*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
˝
save_29/Assign_24Assignvc/dense_2/kernelsave_29/RestoreV2:24*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0*$
_class
loc:@vc/dense_2/kernel
Â
save_29/Assign_25Assignvc/dense_2/kernel/Adamsave_29/RestoreV2:25*
validate_shape(*
T0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
use_locking(
Ä
save_29/Assign_26Assignvc/dense_2/kernel/Adam_1save_29/RestoreV2:26*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	*
validate_shape(
ą
save_29/Assign_27Assignvf/dense/biassave_29/RestoreV2:27*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
ś
save_29/Assign_28Assignvf/dense/bias/Adamsave_29/RestoreV2:28*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
¸
save_29/Assign_29Assignvf/dense/bias/Adam_1save_29/RestoreV2:29*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias
š
save_29/Assign_30Assignvf/dense/kernelsave_29/RestoreV2:30*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	<*
T0*
use_locking(
ž
save_29/Assign_31Assignvf/dense/kernel/Adamsave_29/RestoreV2:31*
T0*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(
Ŕ
save_29/Assign_32Assignvf/dense/kernel/Adam_1save_29/RestoreV2:32*
use_locking(*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
T0
ľ
save_29/Assign_33Assignvf/dense_1/biassave_29/RestoreV2:33*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@vf/dense_1/bias
ş
save_29/Assign_34Assignvf/dense_1/bias/Adamsave_29/RestoreV2:34*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vf/dense_1/bias
ź
save_29/Assign_35Assignvf/dense_1/bias/Adam_1save_29/RestoreV2:35*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias
ž
save_29/Assign_36Assignvf/dense_1/kernelsave_29/RestoreV2:36*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:

Ă
save_29/Assign_37Assignvf/dense_1/kernel/Adamsave_29/RestoreV2:37* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel
Ĺ
save_29/Assign_38Assignvf/dense_1/kernel/Adam_1save_29/RestoreV2:38*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
´
save_29/Assign_39Assignvf/dense_2/biassave_29/RestoreV2:39*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias
š
save_29/Assign_40Assignvf/dense_2/bias/Adamsave_29/RestoreV2:40*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
ť
save_29/Assign_41Assignvf/dense_2/bias/Adam_1save_29/RestoreV2:41*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
˝
save_29/Assign_42Assignvf/dense_2/kernelsave_29/RestoreV2:42*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0
Â
save_29/Assign_43Assignvf/dense_2/kernel/Adamsave_29/RestoreV2:43*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
Ä
save_29/Assign_44Assignvf/dense_2/kernel/Adam_1save_29/RestoreV2:44*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
T0

save_29/restore_shardNoOp^save_29/Assign^save_29/Assign_1^save_29/Assign_10^save_29/Assign_11^save_29/Assign_12^save_29/Assign_13^save_29/Assign_14^save_29/Assign_15^save_29/Assign_16^save_29/Assign_17^save_29/Assign_18^save_29/Assign_19^save_29/Assign_2^save_29/Assign_20^save_29/Assign_21^save_29/Assign_22^save_29/Assign_23^save_29/Assign_24^save_29/Assign_25^save_29/Assign_26^save_29/Assign_27^save_29/Assign_28^save_29/Assign_29^save_29/Assign_3^save_29/Assign_30^save_29/Assign_31^save_29/Assign_32^save_29/Assign_33^save_29/Assign_34^save_29/Assign_35^save_29/Assign_36^save_29/Assign_37^save_29/Assign_38^save_29/Assign_39^save_29/Assign_4^save_29/Assign_40^save_29/Assign_41^save_29/Assign_42^save_29/Assign_43^save_29/Assign_44^save_29/Assign_5^save_29/Assign_6^save_29/Assign_7^save_29/Assign_8^save_29/Assign_9
3
save_29/restore_allNoOp^save_29/restore_shard
\
save_30/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
t
save_30/filenamePlaceholderWithDefaultsave_30/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_30/ConstPlaceholderWithDefaultsave_30/filename*
_output_shapes
: *
dtype0*
shape: 

save_30/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_c15ef1a29de5409292337b551a21e4fa/part*
dtype0
~
save_30/StringJoin
StringJoinsave_30/Constsave_30/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_30/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_30/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 

save_30/ShardedFilenameShardedFilenamesave_30/StringJoinsave_30/ShardedFilename/shardsave_30/num_shards*
_output_shapes
: 
ó
save_30/SaveV2/tensor_namesConst*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
Ŕ
save_30/SaveV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-*
dtype0
ş
save_30/SaveV2SaveV2save_30/ShardedFilenamesave_30/SaveV2/tensor_namessave_30/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_30/control_dependencyIdentitysave_30/ShardedFilename^save_30/SaveV2**
_class 
loc:@save_30/ShardedFilename*
_output_shapes
: *
T0
Ś
.save_30/MergeV2Checkpoints/checkpoint_prefixesPacksave_30/ShardedFilename^save_30/control_dependency*

axis *
T0*
_output_shapes
:*
N

save_30/MergeV2CheckpointsMergeV2Checkpoints.save_30/MergeV2Checkpoints/checkpoint_prefixessave_30/Const*
delete_old_dirs(

save_30/IdentityIdentitysave_30/Const^save_30/MergeV2Checkpoints^save_30/control_dependency*
_output_shapes
: *
T0
ö
save_30/RestoreV2/tensor_namesConst*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
Ă
"save_30/RestoreV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
ű
save_30/RestoreV2	RestoreV2save_30/Constsave_30/RestoreV2/tensor_names"save_30/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
¤
save_30/AssignAssignbeta1_powersave_30/RestoreV2*
T0*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes
: 
¨
save_30/Assign_1Assignbeta2_powersave_30/RestoreV2:1*
use_locking(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
validate_shape(*
T0
Ż
save_30/Assign_2Assignpi/dense/biassave_30/RestoreV2:2* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0
ˇ
save_30/Assign_3Assignpi/dense/kernelsave_30/RestoreV2:3*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
validate_shape(*
T0*
use_locking(
ł
save_30/Assign_4Assignpi/dense_1/biassave_30/RestoreV2:4*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias
ź
save_30/Assign_5Assignpi/dense_1/kernelsave_30/RestoreV2:5*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
˛
save_30/Assign_6Assignpi/dense_2/biassave_30/RestoreV2:6*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias
ť
save_30/Assign_7Assignpi/dense_2/kernelsave_30/RestoreV2:7*
_output_shapes
:	*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(
¨
save_30/Assign_8Assign
pi/log_stdsave_30/RestoreV2:8*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*
_class
loc:@pi/log_std
Ż
save_30/Assign_9Assignvc/dense/biassave_30/RestoreV2:9*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
use_locking(
ś
save_30/Assign_10Assignvc/dense/bias/Adamsave_30/RestoreV2:10*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
T0
¸
save_30/Assign_11Assignvc/dense/bias/Adam_1save_30/RestoreV2:11*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
š
save_30/Assign_12Assignvc/dense/kernelsave_30/RestoreV2:12*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	<
ž
save_30/Assign_13Assignvc/dense/kernel/Adamsave_30/RestoreV2:13*
T0*"
_class
loc:@vc/dense/kernel*
use_locking(*
_output_shapes
:	<*
validate_shape(
Ŕ
save_30/Assign_14Assignvc/dense/kernel/Adam_1save_30/RestoreV2:14*
T0*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vc/dense/kernel*
validate_shape(
ľ
save_30/Assign_15Assignvc/dense_1/biassave_30/RestoreV2:15*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_30/Assign_16Assignvc/dense_1/bias/Adamsave_30/RestoreV2:16*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
ź
save_30/Assign_17Assignvc/dense_1/bias/Adam_1save_30/RestoreV2:17*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias
ž
save_30/Assign_18Assignvc/dense_1/kernelsave_30/RestoreV2:18*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(*
validate_shape(
Ă
save_30/Assign_19Assignvc/dense_1/kernel/Adamsave_30/RestoreV2:19*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:
*
validate_shape(
Ĺ
save_30/Assign_20Assignvc/dense_1/kernel/Adam_1save_30/RestoreV2:20*
T0*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
validate_shape(
´
save_30/Assign_21Assignvc/dense_2/biassave_30/RestoreV2:21*
T0*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_30/Assign_22Assignvc/dense_2/bias/Adamsave_30/RestoreV2:22*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
ť
save_30/Assign_23Assignvc/dense_2/bias/Adam_1save_30/RestoreV2:23*"
_class
loc:@vc/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
˝
save_30/Assign_24Assignvc/dense_2/kernelsave_30/RestoreV2:24*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(*
T0
Â
save_30/Assign_25Assignvc/dense_2/kernel/Adamsave_30/RestoreV2:25*
T0*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(
Ä
save_30/Assign_26Assignvc/dense_2/kernel/Adam_1save_30/RestoreV2:26*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	
ą
save_30/Assign_27Assignvf/dense/biassave_30/RestoreV2:27*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
ś
save_30/Assign_28Assignvf/dense/bias/Adamsave_30/RestoreV2:28*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0*
use_locking(
¸
save_30/Assign_29Assignvf/dense/bias/Adam_1save_30/RestoreV2:29* 
_class
loc:@vf/dense/bias*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(
š
save_30/Assign_30Assignvf/dense/kernelsave_30/RestoreV2:30*
use_locking(*
T0*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
validate_shape(
ž
save_30/Assign_31Assignvf/dense/kernel/Adamsave_30/RestoreV2:31*
_output_shapes
:	<*
use_locking(*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(
Ŕ
save_30/Assign_32Assignvf/dense/kernel/Adam_1save_30/RestoreV2:32*
_output_shapes
:	<*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(
ľ
save_30/Assign_33Assignvf/dense_1/biassave_30/RestoreV2:33*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias
ş
save_30/Assign_34Assignvf/dense_1/bias/Adamsave_30/RestoreV2:34*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(
ź
save_30/Assign_35Assignvf/dense_1/bias/Adam_1save_30/RestoreV2:35*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:
ž
save_30/Assign_36Assignvf/dense_1/kernelsave_30/RestoreV2:36*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(
Ă
save_30/Assign_37Assignvf/dense_1/kernel/Adamsave_30/RestoreV2:37* 
_output_shapes
:
*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(
Ĺ
save_30/Assign_38Assignvf/dense_1/kernel/Adam_1save_30/RestoreV2:38* 
_output_shapes
:
*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0
´
save_30/Assign_39Assignvf/dense_2/biassave_30/RestoreV2:39*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
use_locking(
š
save_30/Assign_40Assignvf/dense_2/bias/Adamsave_30/RestoreV2:40*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(
ť
save_30/Assign_41Assignvf/dense_2/bias/Adam_1save_30/RestoreV2:41*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
˝
save_30/Assign_42Assignvf/dense_2/kernelsave_30/RestoreV2:42*
use_locking(*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0
Â
save_30/Assign_43Assignvf/dense_2/kernel/Adamsave_30/RestoreV2:43*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0
Ä
save_30/Assign_44Assignvf/dense_2/kernel/Adam_1save_30/RestoreV2:44*
_output_shapes
:	*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
T0

save_30/restore_shardNoOp^save_30/Assign^save_30/Assign_1^save_30/Assign_10^save_30/Assign_11^save_30/Assign_12^save_30/Assign_13^save_30/Assign_14^save_30/Assign_15^save_30/Assign_16^save_30/Assign_17^save_30/Assign_18^save_30/Assign_19^save_30/Assign_2^save_30/Assign_20^save_30/Assign_21^save_30/Assign_22^save_30/Assign_23^save_30/Assign_24^save_30/Assign_25^save_30/Assign_26^save_30/Assign_27^save_30/Assign_28^save_30/Assign_29^save_30/Assign_3^save_30/Assign_30^save_30/Assign_31^save_30/Assign_32^save_30/Assign_33^save_30/Assign_34^save_30/Assign_35^save_30/Assign_36^save_30/Assign_37^save_30/Assign_38^save_30/Assign_39^save_30/Assign_4^save_30/Assign_40^save_30/Assign_41^save_30/Assign_42^save_30/Assign_43^save_30/Assign_44^save_30/Assign_5^save_30/Assign_6^save_30/Assign_7^save_30/Assign_8^save_30/Assign_9
3
save_30/restore_allNoOp^save_30/restore_shard
\
save_31/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_31/filenamePlaceholderWithDefaultsave_31/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_31/ConstPlaceholderWithDefaultsave_31/filename*
dtype0*
shape: *
_output_shapes
: 

save_31/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_43651a74f9284f8a9bee39ab594eef97/part
~
save_31/StringJoin
StringJoinsave_31/Constsave_31/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
T
save_31/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_31/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 

save_31/ShardedFilenameShardedFilenamesave_31/StringJoinsave_31/ShardedFilename/shardsave_31/num_shards*
_output_shapes
: 
ó
save_31/SaveV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
Ŕ
save_31/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ş
save_31/SaveV2SaveV2save_31/ShardedFilenamesave_31/SaveV2/tensor_namessave_31/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_31/control_dependencyIdentitysave_31/ShardedFilename^save_31/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_31/ShardedFilename
Ś
.save_31/MergeV2Checkpoints/checkpoint_prefixesPacksave_31/ShardedFilename^save_31/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_31/MergeV2CheckpointsMergeV2Checkpoints.save_31/MergeV2Checkpoints/checkpoint_prefixessave_31/Const*
delete_old_dirs(

save_31/IdentityIdentitysave_31/Const^save_31/MergeV2Checkpoints^save_31/control_dependency*
_output_shapes
: *
T0
ö
save_31/RestoreV2/tensor_namesConst*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-
Ă
"save_31/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ű
save_31/RestoreV2	RestoreV2save_31/Constsave_31/RestoreV2/tensor_names"save_31/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
¤
save_31/AssignAssignbeta1_powersave_31/RestoreV2*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
use_locking(
¨
save_31/Assign_1Assignbeta2_powersave_31/RestoreV2:1*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
validate_shape(*
T0
Ż
save_31/Assign_2Assignpi/dense/biassave_31/RestoreV2:2*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@pi/dense/bias
ˇ
save_31/Assign_3Assignpi/dense/kernelsave_31/RestoreV2:3*
_output_shapes
:	<*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(
ł
save_31/Assign_4Assignpi/dense_1/biassave_31/RestoreV2:4*
T0*
_output_shapes	
:*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(
ź
save_31/Assign_5Assignpi/dense_1/kernelsave_31/RestoreV2:5* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(*
use_locking(
˛
save_31/Assign_6Assignpi/dense_2/biassave_31/RestoreV2:6*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias
ť
save_31/Assign_7Assignpi/dense_2/kernelsave_31/RestoreV2:7*
validate_shape(*
use_locking(*
_output_shapes
:	*$
_class
loc:@pi/dense_2/kernel*
T0
¨
save_31/Assign_8Assign
pi/log_stdsave_31/RestoreV2:8*
validate_shape(*
_class
loc:@pi/log_std*
_output_shapes
:*
T0*
use_locking(
Ż
save_31/Assign_9Assignvc/dense/biassave_31/RestoreV2:9*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
use_locking(*
validate_shape(
ś
save_31/Assign_10Assignvc/dense/bias/Adamsave_31/RestoreV2:10*
validate_shape(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
use_locking(
¸
save_31/Assign_11Assignvc/dense/bias/Adam_1save_31/RestoreV2:11*
T0*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
validate_shape(
š
save_31/Assign_12Assignvc/dense/kernelsave_31/RestoreV2:12*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0
ž
save_31/Assign_13Assignvc/dense/kernel/Adamsave_31/RestoreV2:13*
validate_shape(*
_output_shapes
:	<*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel
Ŕ
save_31/Assign_14Assignvc/dense/kernel/Adam_1save_31/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	<
ľ
save_31/Assign_15Assignvc/dense_1/biassave_31/RestoreV2:15*
use_locking(*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
_output_shapes	
:*
T0
ş
save_31/Assign_16Assignvc/dense_1/bias/Adamsave_31/RestoreV2:16*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias
ź
save_31/Assign_17Assignvc/dense_1/bias/Adam_1save_31/RestoreV2:17*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(*"
_class
loc:@vc/dense_1/bias
ž
save_31/Assign_18Assignvc/dense_1/kernelsave_31/RestoreV2:18*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
*
use_locking(
Ă
save_31/Assign_19Assignvc/dense_1/kernel/Adamsave_31/RestoreV2:19*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0
Ĺ
save_31/Assign_20Assignvc/dense_1/kernel/Adam_1save_31/RestoreV2:20*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
´
save_31/Assign_21Assignvc/dense_2/biassave_31/RestoreV2:21*
T0*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(
š
save_31/Assign_22Assignvc/dense_2/bias/Adamsave_31/RestoreV2:22*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias*
use_locking(*
validate_shape(
ť
save_31/Assign_23Assignvc/dense_2/bias/Adam_1save_31/RestoreV2:23*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0
˝
save_31/Assign_24Assignvc/dense_2/kernelsave_31/RestoreV2:24*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	
Â
save_31/Assign_25Assignvc/dense_2/kernel/Adamsave_31/RestoreV2:25*$
_class
loc:@vc/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	*
validate_shape(
Ä
save_31/Assign_26Assignvc/dense_2/kernel/Adam_1save_31/RestoreV2:26*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
T0*
use_locking(
ą
save_31/Assign_27Assignvf/dense/biassave_31/RestoreV2:27*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias
ś
save_31/Assign_28Assignvf/dense/bias/Adamsave_31/RestoreV2:28*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0*
use_locking(
¸
save_31/Assign_29Assignvf/dense/bias/Adam_1save_31/RestoreV2:29*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(*
T0*
_output_shapes	
:
š
save_31/Assign_30Assignvf/dense/kernelsave_31/RestoreV2:30*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
T0*
use_locking(
ž
save_31/Assign_31Assignvf/dense/kernel/Adamsave_31/RestoreV2:31*"
_class
loc:@vf/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	<
Ŕ
save_31/Assign_32Assignvf/dense/kernel/Adam_1save_31/RestoreV2:32*
_output_shapes
:	<*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(
ľ
save_31/Assign_33Assignvf/dense_1/biassave_31/RestoreV2:33*
use_locking(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0*
validate_shape(
ş
save_31/Assign_34Assignvf/dense_1/bias/Adamsave_31/RestoreV2:34*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:
ź
save_31/Assign_35Assignvf/dense_1/bias/Adam_1save_31/RestoreV2:35*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias
ž
save_31/Assign_36Assignvf/dense_1/kernelsave_31/RestoreV2:36*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:
*
use_locking(
Ă
save_31/Assign_37Assignvf/dense_1/kernel/Adamsave_31/RestoreV2:37*
use_locking(* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
T0
Ĺ
save_31/Assign_38Assignvf/dense_1/kernel/Adam_1save_31/RestoreV2:38*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:

´
save_31/Assign_39Assignvf/dense_2/biassave_31/RestoreV2:39*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
š
save_31/Assign_40Assignvf/dense_2/bias/Adamsave_31/RestoreV2:40*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias
ť
save_31/Assign_41Assignvf/dense_2/bias/Adam_1save_31/RestoreV2:41*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(
˝
save_31/Assign_42Assignvf/dense_2/kernelsave_31/RestoreV2:42*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0
Â
save_31/Assign_43Assignvf/dense_2/kernel/Adamsave_31/RestoreV2:43*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	*
T0
Ä
save_31/Assign_44Assignvf/dense_2/kernel/Adam_1save_31/RestoreV2:44*
validate_shape(*
_output_shapes
:	*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0

save_31/restore_shardNoOp^save_31/Assign^save_31/Assign_1^save_31/Assign_10^save_31/Assign_11^save_31/Assign_12^save_31/Assign_13^save_31/Assign_14^save_31/Assign_15^save_31/Assign_16^save_31/Assign_17^save_31/Assign_18^save_31/Assign_19^save_31/Assign_2^save_31/Assign_20^save_31/Assign_21^save_31/Assign_22^save_31/Assign_23^save_31/Assign_24^save_31/Assign_25^save_31/Assign_26^save_31/Assign_27^save_31/Assign_28^save_31/Assign_29^save_31/Assign_3^save_31/Assign_30^save_31/Assign_31^save_31/Assign_32^save_31/Assign_33^save_31/Assign_34^save_31/Assign_35^save_31/Assign_36^save_31/Assign_37^save_31/Assign_38^save_31/Assign_39^save_31/Assign_4^save_31/Assign_40^save_31/Assign_41^save_31/Assign_42^save_31/Assign_43^save_31/Assign_44^save_31/Assign_5^save_31/Assign_6^save_31/Assign_7^save_31/Assign_8^save_31/Assign_9
3
save_31/restore_allNoOp^save_31/restore_shard
\
save_32/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_32/filenamePlaceholderWithDefaultsave_32/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_32/ConstPlaceholderWithDefaultsave_32/filename*
shape: *
dtype0*
_output_shapes
: 

save_32/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_6bfabe64313f4912bf0ff9c4fc0f43df/part
~
save_32/StringJoin
StringJoinsave_32/Constsave_32/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
T
save_32/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_32/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 

save_32/ShardedFilenameShardedFilenamesave_32/StringJoinsave_32/ShardedFilename/shardsave_32/num_shards*
_output_shapes
: 
ó
save_32/SaveV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
Ŕ
save_32/SaveV2/shape_and_slicesConst*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-
ş
save_32/SaveV2SaveV2save_32/ShardedFilenamesave_32/SaveV2/tensor_namessave_32/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_32/control_dependencyIdentitysave_32/ShardedFilename^save_32/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_32/ShardedFilename
Ś
.save_32/MergeV2Checkpoints/checkpoint_prefixesPacksave_32/ShardedFilename^save_32/control_dependency*

axis *
N*
_output_shapes
:*
T0

save_32/MergeV2CheckpointsMergeV2Checkpoints.save_32/MergeV2Checkpoints/checkpoint_prefixessave_32/Const*
delete_old_dirs(

save_32/IdentityIdentitysave_32/Const^save_32/MergeV2Checkpoints^save_32/control_dependency*
_output_shapes
: *
T0
ö
save_32/RestoreV2/tensor_namesConst*
_output_shapes
:-*
dtype0*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ă
"save_32/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ű
save_32/RestoreV2	RestoreV2save_32/Constsave_32/RestoreV2/tensor_names"save_32/RestoreV2/shape_and_slices*;
dtypes1
/2-*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::
¤
save_32/AssignAssignbeta1_powersave_32/RestoreV2*
use_locking(*
_output_shapes
: *
validate_shape(* 
_class
loc:@vc/dense/bias*
T0
¨
save_32/Assign_1Assignbeta2_powersave_32/RestoreV2:1*
_output_shapes
: *
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
T0
Ż
save_32/Assign_2Assignpi/dense/biassave_32/RestoreV2:2*
use_locking(*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:
ˇ
save_32/Assign_3Assignpi/dense/kernelsave_32/RestoreV2:3*
validate_shape(*
_output_shapes
:	<*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel
ł
save_32/Assign_4Assignpi/dense_1/biassave_32/RestoreV2:4*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:
ź
save_32/Assign_5Assignpi/dense_1/kernelsave_32/RestoreV2:5*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@pi/dense_1/kernel
˛
save_32/Assign_6Assignpi/dense_2/biassave_32/RestoreV2:6*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(
ť
save_32/Assign_7Assignpi/dense_2/kernelsave_32/RestoreV2:7*
T0*
_output_shapes
:	*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel
¨
save_32/Assign_8Assign
pi/log_stdsave_32/RestoreV2:8*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*
_class
loc:@pi/log_std
Ż
save_32/Assign_9Assignvc/dense/biassave_32/RestoreV2:9* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:
ś
save_32/Assign_10Assignvc/dense/bias/Adamsave_32/RestoreV2:10*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(
¸
save_32/Assign_11Assignvc/dense/bias/Adam_1save_32/RestoreV2:11*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
T0*
use_locking(
š
save_32/Assign_12Assignvc/dense/kernelsave_32/RestoreV2:12*
use_locking(*
T0*
_output_shapes
:	<*
validate_shape(*"
_class
loc:@vc/dense/kernel
ž
save_32/Assign_13Assignvc/dense/kernel/Adamsave_32/RestoreV2:13*
T0*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
validate_shape(
Ŕ
save_32/Assign_14Assignvc/dense/kernel/Adam_1save_32/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<*
use_locking(*
T0*
validate_shape(
ľ
save_32/Assign_15Assignvc/dense_1/biassave_32/RestoreV2:15*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias
ş
save_32/Assign_16Assignvc/dense_1/bias/Adamsave_32/RestoreV2:16*
_output_shapes	
:*
T0*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(
ź
save_32/Assign_17Assignvc/dense_1/bias/Adam_1save_32/RestoreV2:17*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
T0
ž
save_32/Assign_18Assignvc/dense_1/kernelsave_32/RestoreV2:18*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
Ă
save_32/Assign_19Assignvc/dense_1/kernel/Adamsave_32/RestoreV2:19*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:

Ĺ
save_32/Assign_20Assignvc/dense_1/kernel/Adam_1save_32/RestoreV2:20*
T0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
validate_shape(
´
save_32/Assign_21Assignvc/dense_2/biassave_32/RestoreV2:21*
T0*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(*
validate_shape(
š
save_32/Assign_22Assignvc/dense_2/bias/Adamsave_32/RestoreV2:22*"
_class
loc:@vc/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
ť
save_32/Assign_23Assignvc/dense_2/bias/Adam_1save_32/RestoreV2:23*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(*
T0
˝
save_32/Assign_24Assignvc/dense_2/kernelsave_32/RestoreV2:24*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	
Â
save_32/Assign_25Assignvc/dense_2/kernel/Adamsave_32/RestoreV2:25*
validate_shape(*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
use_locking(
Ä
save_32/Assign_26Assignvc/dense_2/kernel/Adam_1save_32/RestoreV2:26*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	*
T0
ą
save_32/Assign_27Assignvf/dense/biassave_32/RestoreV2:27*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0* 
_class
loc:@vf/dense/bias
ś
save_32/Assign_28Assignvf/dense/bias/Adamsave_32/RestoreV2:28*
use_locking(* 
_class
loc:@vf/dense/bias*
T0*
validate_shape(*
_output_shapes	
:
¸
save_32/Assign_29Assignvf/dense/bias/Adam_1save_32/RestoreV2:29*
use_locking(*
T0*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
š
save_32/Assign_30Assignvf/dense/kernelsave_32/RestoreV2:30*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes
:	<*
T0
ž
save_32/Assign_31Assignvf/dense/kernel/Adamsave_32/RestoreV2:31*
_output_shapes
:	<*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense/kernel*
T0
Ŕ
save_32/Assign_32Assignvf/dense/kernel/Adam_1save_32/RestoreV2:32*
_output_shapes
:	<*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(*
use_locking(
ľ
save_32/Assign_33Assignvf/dense_1/biassave_32/RestoreV2:33*
validate_shape(*
use_locking(*
_output_shapes	
:*"
_class
loc:@vf/dense_1/bias*
T0
ş
save_32/Assign_34Assignvf/dense_1/bias/Adamsave_32/RestoreV2:34*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(
ź
save_32/Assign_35Assignvf/dense_1/bias/Adam_1save_32/RestoreV2:35*
_output_shapes	
:*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
validate_shape(
ž
save_32/Assign_36Assignvf/dense_1/kernelsave_32/RestoreV2:36*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:

Ă
save_32/Assign_37Assignvf/dense_1/kernel/Adamsave_32/RestoreV2:37* 
_output_shapes
:
*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(
Ĺ
save_32/Assign_38Assignvf/dense_1/kernel/Adam_1save_32/RestoreV2:38*
T0* 
_output_shapes
:
*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(
´
save_32/Assign_39Assignvf/dense_2/biassave_32/RestoreV2:39*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0*
use_locking(*
validate_shape(
š
save_32/Assign_40Assignvf/dense_2/bias/Adamsave_32/RestoreV2:40*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:
ť
save_32/Assign_41Assignvf/dense_2/bias/Adam_1save_32/RestoreV2:41*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias
˝
save_32/Assign_42Assignvf/dense_2/kernelsave_32/RestoreV2:42*$
_class
loc:@vf/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	*
validate_shape(
Â
save_32/Assign_43Assignvf/dense_2/kernel/Adamsave_32/RestoreV2:43*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(
Ä
save_32/Assign_44Assignvf/dense_2/kernel/Adam_1save_32/RestoreV2:44*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
use_locking(

save_32/restore_shardNoOp^save_32/Assign^save_32/Assign_1^save_32/Assign_10^save_32/Assign_11^save_32/Assign_12^save_32/Assign_13^save_32/Assign_14^save_32/Assign_15^save_32/Assign_16^save_32/Assign_17^save_32/Assign_18^save_32/Assign_19^save_32/Assign_2^save_32/Assign_20^save_32/Assign_21^save_32/Assign_22^save_32/Assign_23^save_32/Assign_24^save_32/Assign_25^save_32/Assign_26^save_32/Assign_27^save_32/Assign_28^save_32/Assign_29^save_32/Assign_3^save_32/Assign_30^save_32/Assign_31^save_32/Assign_32^save_32/Assign_33^save_32/Assign_34^save_32/Assign_35^save_32/Assign_36^save_32/Assign_37^save_32/Assign_38^save_32/Assign_39^save_32/Assign_4^save_32/Assign_40^save_32/Assign_41^save_32/Assign_42^save_32/Assign_43^save_32/Assign_44^save_32/Assign_5^save_32/Assign_6^save_32/Assign_7^save_32/Assign_8^save_32/Assign_9
3
save_32/restore_allNoOp^save_32/restore_shard
\
save_33/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_33/filenamePlaceholderWithDefaultsave_33/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_33/ConstPlaceholderWithDefaultsave_33/filename*
_output_shapes
: *
shape: *
dtype0

save_33/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_596815d362004c4f9ec3585e139e438b/part*
dtype0
~
save_33/StringJoin
StringJoinsave_33/Constsave_33/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_33/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
_
save_33/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_33/ShardedFilenameShardedFilenamesave_33/StringJoinsave_33/ShardedFilename/shardsave_33/num_shards*
_output_shapes
: 
ó
save_33/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ŕ
save_33/SaveV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-*
dtype0
ş
save_33/SaveV2SaveV2save_33/ShardedFilenamesave_33/SaveV2/tensor_namessave_33/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_33/control_dependencyIdentitysave_33/ShardedFilename^save_33/SaveV2**
_class 
loc:@save_33/ShardedFilename*
_output_shapes
: *
T0
Ś
.save_33/MergeV2Checkpoints/checkpoint_prefixesPacksave_33/ShardedFilename^save_33/control_dependency*
N*

axis *
_output_shapes
:*
T0

save_33/MergeV2CheckpointsMergeV2Checkpoints.save_33/MergeV2Checkpoints/checkpoint_prefixessave_33/Const*
delete_old_dirs(

save_33/IdentityIdentitysave_33/Const^save_33/MergeV2Checkpoints^save_33/control_dependency*
_output_shapes
: *
T0
ö
save_33/RestoreV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
Ă
"save_33/RestoreV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-*
dtype0
ű
save_33/RestoreV2	RestoreV2save_33/Constsave_33/RestoreV2/tensor_names"save_33/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
¤
save_33/AssignAssignbeta1_powersave_33/RestoreV2*
use_locking(*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias*
validate_shape(
¨
save_33/Assign_1Assignbeta2_powersave_33/RestoreV2:1*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: 
Ż
save_33/Assign_2Assignpi/dense/biassave_33/RestoreV2:2* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(
ˇ
save_33/Assign_3Assignpi/dense/kernelsave_33/RestoreV2:3*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	<*
validate_shape(
ł
save_33/Assign_4Assignpi/dense_1/biassave_33/RestoreV2:4*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_33/Assign_5Assignpi/dense_1/kernelsave_33/RestoreV2:5* 
_output_shapes
:
*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel
˛
save_33/Assign_6Assignpi/dense_2/biassave_33/RestoreV2:6*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
ť
save_33/Assign_7Assignpi/dense_2/kernelsave_33/RestoreV2:7*
T0*
use_locking(*
_output_shapes
:	*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
¨
save_33/Assign_8Assign
pi/log_stdsave_33/RestoreV2:8*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*
_class
loc:@pi/log_std
Ż
save_33/Assign_9Assignvc/dense/biassave_33/RestoreV2:9*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias
ś
save_33/Assign_10Assignvc/dense/bias/Adamsave_33/RestoreV2:10*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(* 
_class
loc:@vc/dense/bias
¸
save_33/Assign_11Assignvc/dense/bias/Adam_1save_33/RestoreV2:11* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
š
save_33/Assign_12Assignvc/dense/kernelsave_33/RestoreV2:12*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<
ž
save_33/Assign_13Assignvc/dense/kernel/Adamsave_33/RestoreV2:13*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	<*
use_locking(
Ŕ
save_33/Assign_14Assignvc/dense/kernel/Adam_1save_33/RestoreV2:14*
use_locking(*
validate_shape(*
_output_shapes
:	<*"
_class
loc:@vc/dense/kernel*
T0
ľ
save_33/Assign_15Assignvc/dense_1/biassave_33/RestoreV2:15*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
ş
save_33/Assign_16Assignvc/dense_1/bias/Adamsave_33/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:*
validate_shape(
ź
save_33/Assign_17Assignvc/dense_1/bias/Adam_1save_33/RestoreV2:17*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
ž
save_33/Assign_18Assignvc/dense_1/kernelsave_33/RestoreV2:18*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0*$
_class
loc:@vc/dense_1/kernel
Ă
save_33/Assign_19Assignvc/dense_1/kernel/Adamsave_33/RestoreV2:19*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:

Ĺ
save_33/Assign_20Assignvc/dense_1/kernel/Adam_1save_33/RestoreV2:20*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel
´
save_33/Assign_21Assignvc/dense_2/biassave_33/RestoreV2:21*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
š
save_33/Assign_22Assignvc/dense_2/bias/Adamsave_33/RestoreV2:22*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
ť
save_33/Assign_23Assignvc/dense_2/bias/Adam_1save_33/RestoreV2:23*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
˝
save_33/Assign_24Assignvc/dense_2/kernelsave_33/RestoreV2:24*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(
Â
save_33/Assign_25Assignvc/dense_2/kernel/Adamsave_33/RestoreV2:25*
_output_shapes
:	*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(*
T0
Ä
save_33/Assign_26Assignvc/dense_2/kernel/Adam_1save_33/RestoreV2:26*$
_class
loc:@vc/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	*
use_locking(
ą
save_33/Assign_27Assignvf/dense/biassave_33/RestoreV2:27*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ś
save_33/Assign_28Assignvf/dense/bias/Adamsave_33/RestoreV2:28*
validate_shape(*
T0*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
use_locking(
¸
save_33/Assign_29Assignvf/dense/bias/Adam_1save_33/RestoreV2:29*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:
š
save_33/Assign_30Assignvf/dense/kernelsave_33/RestoreV2:30*
_output_shapes
:	<*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
use_locking(
ž
save_33/Assign_31Assignvf/dense/kernel/Adamsave_33/RestoreV2:31*
_output_shapes
:	<*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(
Ŕ
save_33/Assign_32Assignvf/dense/kernel/Adam_1save_33/RestoreV2:32*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<*
validate_shape(*
T0*
use_locking(
ľ
save_33/Assign_33Assignvf/dense_1/biassave_33/RestoreV2:33*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0
ş
save_33/Assign_34Assignvf/dense_1/bias/Adamsave_33/RestoreV2:34*
_output_shapes	
:*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(
ź
save_33/Assign_35Assignvf/dense_1/bias/Adam_1save_33/RestoreV2:35*"
_class
loc:@vf/dense_1/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:
ž
save_33/Assign_36Assignvf/dense_1/kernelsave_33/RestoreV2:36*
T0*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
use_locking(
Ă
save_33/Assign_37Assignvf/dense_1/kernel/Adamsave_33/RestoreV2:37*
validate_shape(* 
_output_shapes
:
*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel
Ĺ
save_33/Assign_38Assignvf/dense_1/kernel/Adam_1save_33/RestoreV2:38*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
´
save_33/Assign_39Assignvf/dense_2/biassave_33/RestoreV2:39*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
š
save_33/Assign_40Assignvf/dense_2/bias/Adamsave_33/RestoreV2:40*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
ť
save_33/Assign_41Assignvf/dense_2/bias/Adam_1save_33/RestoreV2:41*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
˝
save_33/Assign_42Assignvf/dense_2/kernelsave_33/RestoreV2:42*
_output_shapes
:	*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
Â
save_33/Assign_43Assignvf/dense_2/kernel/Adamsave_33/RestoreV2:43*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	
Ä
save_33/Assign_44Assignvf/dense_2/kernel/Adam_1save_33/RestoreV2:44*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
T0*
validate_shape(*
use_locking(

save_33/restore_shardNoOp^save_33/Assign^save_33/Assign_1^save_33/Assign_10^save_33/Assign_11^save_33/Assign_12^save_33/Assign_13^save_33/Assign_14^save_33/Assign_15^save_33/Assign_16^save_33/Assign_17^save_33/Assign_18^save_33/Assign_19^save_33/Assign_2^save_33/Assign_20^save_33/Assign_21^save_33/Assign_22^save_33/Assign_23^save_33/Assign_24^save_33/Assign_25^save_33/Assign_26^save_33/Assign_27^save_33/Assign_28^save_33/Assign_29^save_33/Assign_3^save_33/Assign_30^save_33/Assign_31^save_33/Assign_32^save_33/Assign_33^save_33/Assign_34^save_33/Assign_35^save_33/Assign_36^save_33/Assign_37^save_33/Assign_38^save_33/Assign_39^save_33/Assign_4^save_33/Assign_40^save_33/Assign_41^save_33/Assign_42^save_33/Assign_43^save_33/Assign_44^save_33/Assign_5^save_33/Assign_6^save_33/Assign_7^save_33/Assign_8^save_33/Assign_9
3
save_33/restore_allNoOp^save_33/restore_shard
\
save_34/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
t
save_34/filenamePlaceholderWithDefaultsave_34/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_34/ConstPlaceholderWithDefaultsave_34/filename*
dtype0*
_output_shapes
: *
shape: 

save_34/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2f7ea326029f4c29a92e2cd69d30d4fe/part
~
save_34/StringJoin
StringJoinsave_34/Constsave_34/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_34/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_34/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_34/ShardedFilenameShardedFilenamesave_34/StringJoinsave_34/ShardedFilename/shardsave_34/num_shards*
_output_shapes
: 
ó
save_34/SaveV2/tensor_namesConst*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
Ŕ
save_34/SaveV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:-
ş
save_34/SaveV2SaveV2save_34/ShardedFilenamesave_34/SaveV2/tensor_namessave_34/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-

save_34/control_dependencyIdentitysave_34/ShardedFilename^save_34/SaveV2*
_output_shapes
: **
_class 
loc:@save_34/ShardedFilename*
T0
Ś
.save_34/MergeV2Checkpoints/checkpoint_prefixesPacksave_34/ShardedFilename^save_34/control_dependency*

axis *
_output_shapes
:*
T0*
N

save_34/MergeV2CheckpointsMergeV2Checkpoints.save_34/MergeV2Checkpoints/checkpoint_prefixessave_34/Const*
delete_old_dirs(

save_34/IdentityIdentitysave_34/Const^save_34/MergeV2Checkpoints^save_34/control_dependency*
_output_shapes
: *
T0
ö
save_34/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:-*Ł
valueB-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
Ă
"save_34/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ű
save_34/RestoreV2	RestoreV2save_34/Constsave_34/RestoreV2/tensor_names"save_34/RestoreV2/shape_and_slices*Ę
_output_shapesˇ
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
¤
save_34/AssignAssignbeta1_powersave_34/RestoreV2*
use_locking(*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0*
validate_shape(
¨
save_34/Assign_1Assignbeta2_powersave_34/RestoreV2:1* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: *
T0
Ż
save_34/Assign_2Assignpi/dense/biassave_34/RestoreV2:2*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:*
T0*
use_locking(
ˇ
save_34/Assign_3Assignpi/dense/kernelsave_34/RestoreV2:3*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<*
T0
ł
save_34/Assign_4Assignpi/dense_1/biassave_34/RestoreV2:4*
use_locking(*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(
ź
save_34/Assign_5Assignpi/dense_1/kernelsave_34/RestoreV2:5*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
*
validate_shape(*
use_locking(
˛
save_34/Assign_6Assignpi/dense_2/biassave_34/RestoreV2:6*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
ť
save_34/Assign_7Assignpi/dense_2/kernelsave_34/RestoreV2:7*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
:	
¨
save_34/Assign_8Assign
pi/log_stdsave_34/RestoreV2:8*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*
_class
loc:@pi/log_std
Ż
save_34/Assign_9Assignvc/dense/biassave_34/RestoreV2:9*
use_locking(*
validate_shape(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0
ś
save_34/Assign_10Assignvc/dense/bias/Adamsave_34/RestoreV2:10*
validate_shape(*
use_locking(*
_output_shapes	
:* 
_class
loc:@vc/dense/bias*
T0
¸
save_34/Assign_11Assignvc/dense/bias/Adam_1save_34/RestoreV2:11*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes	
:*
validate_shape(*
T0
š
save_34/Assign_12Assignvc/dense/kernelsave_34/RestoreV2:12*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<*
validate_shape(
ž
save_34/Assign_13Assignvc/dense/kernel/Adamsave_34/RestoreV2:13*
_output_shapes
:	<*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(
Ŕ
save_34/Assign_14Assignvc/dense/kernel/Adam_1save_34/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	<*
use_locking(
ľ
save_34/Assign_15Assignvc/dense_1/biassave_34/RestoreV2:15*
T0*
use_locking(*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(
ş
save_34/Assign_16Assignvc/dense_1/bias/Adamsave_34/RestoreV2:16*
use_locking(*
T0*
_output_shapes	
:*"
_class
loc:@vc/dense_1/bias*
validate_shape(
ź
save_34/Assign_17Assignvc/dense_1/bias/Adam_1save_34/RestoreV2:17*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:
ž
save_34/Assign_18Assignvc/dense_1/kernelsave_34/RestoreV2:18*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel
Ă
save_34/Assign_19Assignvc/dense_1/kernel/Adamsave_34/RestoreV2:19*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:

Ĺ
save_34/Assign_20Assignvc/dense_1/kernel/Adam_1save_34/RestoreV2:20* 
_output_shapes
:
*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
T0*
validate_shape(
´
save_34/Assign_21Assignvc/dense_2/biassave_34/RestoreV2:21*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0
š
save_34/Assign_22Assignvc/dense_2/bias/Adamsave_34/RestoreV2:22*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
T0
ť
save_34/Assign_23Assignvc/dense_2/bias/Adam_1save_34/RestoreV2:23*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
˝
save_34/Assign_24Assignvc/dense_2/kernelsave_34/RestoreV2:24*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0
Â
save_34/Assign_25Assignvc/dense_2/kernel/Adamsave_34/RestoreV2:25*
use_locking(*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	*
validate_shape(
Ä
save_34/Assign_26Assignvc/dense_2/kernel/Adam_1save_34/RestoreV2:26*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(
ą
save_34/Assign_27Assignvf/dense/biassave_34/RestoreV2:27*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
_output_shapes	
:
ś
save_34/Assign_28Assignvf/dense/bias/Adamsave_34/RestoreV2:28*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(* 
_class
loc:@vf/dense/bias
¸
save_34/Assign_29Assignvf/dense/bias/Adam_1save_34/RestoreV2:29*
T0*
use_locking(*
_output_shapes	
:* 
_class
loc:@vf/dense/bias*
validate_shape(
š
save_34/Assign_30Assignvf/dense/kernelsave_34/RestoreV2:30*
use_locking(*
_output_shapes
:	<*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel
ž
save_34/Assign_31Assignvf/dense/kernel/Adamsave_34/RestoreV2:31*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<
Ŕ
save_34/Assign_32Assignvf/dense/kernel/Adam_1save_34/RestoreV2:32*"
_class
loc:@vf/dense/kernel*
use_locking(*
T0*
_output_shapes
:	<*
validate_shape(
ľ
save_34/Assign_33Assignvf/dense_1/biassave_34/RestoreV2:33*
T0*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:
ş
save_34/Assign_34Assignvf/dense_1/bias/Adamsave_34/RestoreV2:34*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes	
:
ź
save_34/Assign_35Assignvf/dense_1/bias/Adam_1save_34/RestoreV2:35*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:
ž
save_34/Assign_36Assignvf/dense_1/kernelsave_34/RestoreV2:36*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*$
_class
loc:@vf/dense_1/kernel
Ă
save_34/Assign_37Assignvf/dense_1/kernel/Adamsave_34/RestoreV2:37*
T0* 
_output_shapes
:
*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel
Ĺ
save_34/Assign_38Assignvf/dense_1/kernel/Adam_1save_34/RestoreV2:38*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
´
save_34/Assign_39Assignvf/dense_2/biassave_34/RestoreV2:39*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
š
save_34/Assign_40Assignvf/dense_2/bias/Adamsave_34/RestoreV2:40*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
use_locking(
ť
save_34/Assign_41Assignvf/dense_2/bias/Adam_1save_34/RestoreV2:41*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0*
use_locking(
˝
save_34/Assign_42Assignvf/dense_2/kernelsave_34/RestoreV2:42*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(
Â
save_34/Assign_43Assignvf/dense_2/kernel/Adamsave_34/RestoreV2:43*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	*
T0*
use_locking(
Ä
save_34/Assign_44Assignvf/dense_2/kernel/Adam_1save_34/RestoreV2:44*
validate_shape(*
T0*
_output_shapes
:	*$
_class
loc:@vf/dense_2/kernel*
use_locking(

save_34/restore_shardNoOp^save_34/Assign^save_34/Assign_1^save_34/Assign_10^save_34/Assign_11^save_34/Assign_12^save_34/Assign_13^save_34/Assign_14^save_34/Assign_15^save_34/Assign_16^save_34/Assign_17^save_34/Assign_18^save_34/Assign_19^save_34/Assign_2^save_34/Assign_20^save_34/Assign_21^save_34/Assign_22^save_34/Assign_23^save_34/Assign_24^save_34/Assign_25^save_34/Assign_26^save_34/Assign_27^save_34/Assign_28^save_34/Assign_29^save_34/Assign_3^save_34/Assign_30^save_34/Assign_31^save_34/Assign_32^save_34/Assign_33^save_34/Assign_34^save_34/Assign_35^save_34/Assign_36^save_34/Assign_37^save_34/Assign_38^save_34/Assign_39^save_34/Assign_4^save_34/Assign_40^save_34/Assign_41^save_34/Assign_42^save_34/Assign_43^save_34/Assign_44^save_34/Assign_5^save_34/Assign_6^save_34/Assign_7^save_34/Assign_8^save_34/Assign_9
3
save_34/restore_allNoOp^save_34/restore_shard "E
save_34/Const:0save_34/Identity:0save_34/restore_all (5 @F8"
train_op

Adam"đ
trainable_variablesŘŐ
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
vc/dense_2/bias:0vc/dense_2/bias/Assignvc/dense_2/bias/read:02#vc/dense_2/bias/Initializer/zeros:08"đ*
	variablesâ*ß*
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
vc/dense_2/bias/Adam_1:0vc/dense_2/bias/Adam_1/Assignvc/dense_2/bias/Adam_1/read:02*vc/dense_2/bias/Adam_1/Initializer/zeros:0*Ď
serving_defaultť
)
x$
Placeholder:0˙˙˙˙˙˙˙˙˙<%
vc
vc/Squeeze:0˙˙˙˙˙˙˙˙˙%
pi
pi/add:0˙˙˙˙˙˙˙˙˙$
v
vf/Squeeze:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict