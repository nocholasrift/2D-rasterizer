from PIL import Image
import sys
import math

f = str(sys.argv[1])
file = open(f, "r")
lines = file.readlines()
file.close()

vertices = []
words = lines[0].split(" ")
filename = words[3].strip()

img = Image.new("RGBA", (int(words[1]), int(words[2])), (0,0,0,0))
img.save(filename)
putpixel = img.im.putpixel

color_vals = [255,255,255]
transformation = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
projection = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
#used for viewport transformations
width = int(words[1])
height = int(words[2])
cull = False

z_buffer = [[1.0 for i in range(height)] for i in range(width)]

def input(line):
	global cull
	global transformation
	if line[:3]=="xyz" and line[3]!="w":
		words_init = line.split(" ")
		words =[]
		for word in words_init:
			if (word != ""):
				words.append(word)
		col = list(color_vals)
		points = (float(words[1]),float(words[2]),float(words[3]),1,col)
		vertices.append(points)
	if line[:4]=="xyzw":
		words_init = line.split(" ")
		words =[]
		for word in words_init:
			if (word != ""):
				words.append(word)
		col = list(color_vals)
		points = (float(words[1]),float(words[2]),float(words[3]),float(words[4]),col)
		vertices.append(points)
	if line[:4]=="trif":
		words_init = line.split(" ")
		words =[]
		for word in words_init:
			if (word != ""):
				words.append(word)
		words[1] = int(words[1]) if int(words[1]) < 0 else int(words[1]) - 1
		words[2] = int(words[2]) if int(words[2]) < 0 else int(words[2]) - 1
		words[3] = int(words[3]) if int(words[3]) < 0 else int(words[3]) - 1
		#untransformed points defining the triangle
		untrans_points = (vertices[words[1]], vertices[words[2]], vertices[words[3]])
		#transformed points
		trans_points = []
		for point in untrans_points:
			trans_points.append(apply_matrix_trans(point))
		proj_points = []
		for point in trans_points:
			proj_points.append(apply_proj_trans(point))

		if(cull and is_counter_clockwise(proj_points)):
			trif(proj_points, False)
		elif (not cull):
			trif(proj_points, False)
	if line[:5]=="color":
		words_init = line.split(" ")
		words =[]
		for word in words_init:
			if (word != ""):
				words.append(word)
		color(float(words[1]),float(words[2]),float(words[3]))		
	if line[:6]=="loadmv":
		words_init = line.split(" ")
		words =[]
		i,j=0,0
		for word in words_init:
			if (word != "" and word != "loadmv"):
				transformation[j][i]=float(word)
				i=i+1
				j=j if i < 4 else j+1
				i=i if i < 4 else 0
	if line[:5]=="loadp":
		words_init = line.split(" ")
		words =[]
		i,j=0,0
		for word in words_init:
			if (word != "" and word != "loadp"):
				projection[j][i]=float(word)
				i=i+1
				j=j if i < 4 else j+1
				i=i if i < 4 else 0
	if line[:7]=="frustum":
		words_init = line.split(" ")
		words =[]
		for word in words_init:
			if (word != "" and word != "frustum"):
				words.append(float(word))
		for i in range(4):
			for j in range(4):
				projection[i][j] = 0.0
		l,r,b,t,n,f = words[0],words[1],words[2],words[3],words[4],words[5]
		a_m,b_m,c_m,d_m = round((r+l)/(r-l),5),round((t+b)/(t-b),5),round(-1*(f+n)/(f-n),5),round(-2*(f*n)/(f-n),5)
		projection[0][0] = round(2*n/(r-l),5)
		projection[1][1] = round(2*n/(t-b),5)
		projection[0][2],projection[1][2],projection[2][2],projection[2][3] = a_m,b_m,c_m,d_m
		projection[3][2] = -1
	if line[:5]=="ortho":
		words_init = line.split(" ")
		words =[]
		for word in words_init:
			if (word != "" and word != "ortho"):
				words.append(float(word))
		for i in range(4):
			for j in range(4):
				projection[i][j] = 0.0
		l,r,b,t,n,f = words[0],words[1],words[2],words[3],words[4],words[5]	
		n=2*n - f	
		tx,ty,tz=round(-1*(r+l)/(r-l),5),round(-1*(t+b)/(t-b),5),round(-1*(f+n)/(f-n),5)
		projection[0][0],projection[1][1],projection[2][2],projection[3][3] = round(2/(r-l),5),round(2/(t-b),5),round(-2/(f-n),5),1
		projection[0][3],projection[1][3],projection[2][3] = tx,ty,tz
	if line[:9]=="translate":
		words_init = line.split(" ")
		words =[]
		for word in words_init:
			if (word != "" and word != "translate"):
				words.append(float(word))
		for i in range(3):
			transformation[i][3]+=words[i]
	if line[:5]=="scale":
		words_init = line.split(" ")
		words =[]
		for word in words_init:
			if (word != "" and word != "scale"):
				words.append(float(word))
		for i in range(3):
			for j in range(i,i+1):
				transformation[i][j]*=words[i]
	if line[:4]=="cull":
		cull = True
	if line[:6]=="lookat":
		words_init = line.split(" ")
		words =[]
		for word in words_init:
			if (word != "" and word != "lookat"):
				words.append(float(word))
		for i in range(4):
			for j in range(4):
				transformation[i][j] = 0.0
		words[0] = words[0] if words[0] < 0 else words[0] - 1
		words[1] = words[1] if words[1] < 0 else words[1] - 1
		eye,center,up=vertices[int(words[0])],vertices[int(words[1])],[words[2],words[3],words[4]]
		f=normalize([center[0]-eye[0],center[1]-eye[1],center[2]-eye[2]])
		up=normalize(up)
		s=cross_product(f,up)
		s_norm = normalize(s)
		u=cross_product(s_norm,f)
		transformation=[[s[0],s[1],s[2],0],[u[0],u[1],u[2],0],[-1*f[0],-1*f[1],-1*f[2],0],[0,0,0,1]]
		#transformation[0][0],transformation[0][1],transformation[0][2]=s[0],s[1],s[2]
		#transformation[1][0],transformation[1][1],transformation[1][2]=u[0],u[1],u[2]
		#transformation[2][0],transformation[2][1],transformation[2][2],transformation[3][3]=-1*f[0],-1*f[1],-1*f[2],1
		translation = [[1,0,0,-1*eye[0]],[0,1,0,-1*eye[1]],[0,0,1,-1*eye[2]],[0,0,0,1]]
		transformation = mult4by4(transformation, translation)
	if line[:4]=="trig":
		words_init = line.split(" ")
		words =[]
		for word in words_init:
			if (word != ""):
				words.append(word)
		words[1] = int(words[1]) if int(words[1]) < 0 else int(words[1]) - 1
		words[2] = int(words[2]) if int(words[2]) < 0 else int(words[2]) - 1
		words[3] = int(words[3]) if int(words[3]) < 0 else int(words[3]) - 1
		#untransformed points defining the triangle
		untrans_points = (vertices[words[1]], vertices[words[2]], vertices[words[3]])
		#transformed points
		trans_points = []
		for point in untrans_points:
			trans_points.append(apply_matrix_trans(point))
		proj_points = []
		for point in trans_points:
			proj_points.append(apply_proj_trans(point))
		
		if(cull and is_counter_clockwise(proj_points)):
			trif(proj_points, True)
		elif (not cull):
			trif(proj_points, True)
	if line[:6]=="rotate" and line[6]==" ":
		words_init = line.split(" ")
		words =[]
		for word in words_init:
			if (word != "" and word != "rotate"):
				words.append(float(word))
		c,s = math.cos(math.radians(words[0])),math.sin(math.radians(words[0]))
		p=[words[1],words[2],words[3]]
		p=normalize(p)
		x,y,z=p[0],p[1],p[2]
		transformation=mult4by4(transformation,[[x**2*(1-c) +c,x*y*(1-c)-(z*s),x*z*(1-c)+(y*s),0],[y*x*(1-c) +(z*s),y**2*(1-c)+c,y*z*(1-c)-(x*s),0],[x*z*(1-c)-(y*s),y*z*(1-c)+(x*s),z**2*(1-c)+c,0],[0,0,0,1]])
	if line[:7]=="rotatex":
		words_init = line.split(" ")
		words =[]
		for word in words_init:
			if (word != "" and word != "rotatex"):
				words.append(float(word))
		c,s=math.cos(math.radians(words[0])),math.sin(math.radians(words[0]))
		transformation=mult4by4(transformation,[[1,0,0,0],[0,c,-1*s,0],[0,s,c,0],[0,0,0,1]])
	if line[:7]=="rotatey":
		words_init = line.split(" ")
		words =[]
		for word in words_init:
			if (word != "" and word != "rotatey"):
				words.append(float(word))
		c,s=math.cos(math.radians(words[0])),math.sin(math.radians(words[0]))
		transformation=mult4by4(transformation,[[c,0,s,0],[0,1,0,0],[-1*s,0,c,0],[0,0,0,1]])
	if line[:7]=="rotatez":
		words_init = line.split(" ")
		words =[]
		for word in words_init:
			if (word != "" and word != "rotatez"):
				words.append(float(word))
		c,s=math.cos(math.radians(words[0])),math.sin(math.radians(words[0]))
		transformation=mult4by4(transformation,[[c,-1*s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])

def color(r, g, b):
	color_vals[0] = int(r*255) if r >=0 else 0
	color_vals[1] = int(g*255) if g >=0 else 0
	color_vals[2] = int(b*255) if b >=0 else 0

def linec(points, ystep, col_interp,draw):
	edges = []
	p1 = points[0]
	p2 = points[1]

	_len = p1[0] - p2[0] if (abs(p1[0] - p2[0]) > abs(p1[1] - p2[1]) and not ystep) else p1[1] - p2[1]

	start,end = p1,p2
	if(_len < 0):
		_len = _len*-1

	x_step,y_step = False,False

	if(abs(end[0]-start[0]) > abs(end[1]-start[1]) and not ystep):
		x_step = True
	else:
		y_step = True
	
	if(x_step and end[0] < start[0]):
		start,end = end,start
	if(y_step and end[1] < start[1]):
		start,end = end,start
	#change in x,y,z,red,green,blue
	qz,qy,qx=0,0,0
	if(_len!=0):
		qx,qy,qz = (end[0]-start[0])/_len,(end[1]-start[1])/_len,(end[2]-start[2])/_len

	qr,qg,qb,qa=0,0,0,0
	if(col_interp and _len !=0):
		qr,qg,qb,qa = (end[4][0]-start[4][0])/_len,(end[4][1]-start[4][1])/_len,(end[4][2]-start[4][2])/_len,0
	
	#initializing x,y,r,g,b,a
	x,y,z= start[0],start[1],start[2]

	if(col_interp):
		r,g,b,a=start[4][0],start[4][1],start[4][2],255
	else:
		r,g,b,a=color_vals[0],color_vals[1],color_vals[2],255
		#r,g,b,a=start[2][0],start[2][1],start[2][2],start[2][3]

	rx,ry,rz,rr,rg,rb,ra=0,0,0,0,0,0,0

	if(x_step and start[0]//1 != start[0]): #if step in x
		off =(start[0]+1)//1 - start[0]
		rx,ry,rz,rr,rg,rb,ra = round(qx*off,10),round(qy*off,10),round(qz*off,10),qr*off,qg*off,qb*off,qa*off
	elif(y_step and start[1]//1 != start[1]):
		off = (start[1]+1)//1 - start[1]
		rx,ry,rz,rr,rg,rb,ra = round(qx*off,10),round(qy*off,10),round(qz*off,10),qr*off,qg*off,qb*off,qa*off

	x = x + rx
	y = y + ry
	z = z + rz
	r,g,b,a = r +rr,g+rg,b+rb,a+ra

	i,j,i_max,di,dj = 0,0,0,0,0
	if(x_step):
		i,j,di,dj,i_max = x,y,qx,qy,end[0]
	else:
		i,j,di,dj,i_max = y,x,qy,qx,end[1]
	#print(start)
	while(i < i_max):
		if(x_step):
			#print(int(i+.5),j)
			if(int(i+.5) < i_max and round(i) >= 0 and round(i) < width and j >= 0 and round(j) < height and draw and round(z,5) >0 and round(z,5) <1 and z_buffer[int(i+.5)][int(j+.5)] > z):
				putpixel((int(i+.5),int((j+.5))),(int(r),int(g),int(b),int(a)))
				z_buffer[int(i+.5)][int(j+.5)] = z
			elif(not draw):
				edges.append(((i,j,z,1),(r,g,b,a)))
			
		else:
			if(int(i+.5) < i_max and round(i) >= 0 and round(i) < height and j >= 0 and round(j) < width and draw and round(z,5) >0 and round(z,5) <1 and z_buffer[int(j+.5)][int(i+.5)] > z):
				#print(j,i)
				putpixel((int((j+.5)),int(i+.5)),(int(r),int(g),int(b),int(a)))
				z_buffer[int(j+.5)][int(i+.5)] = z
			elif( int(z) >=0 and int(z) <=1 and not draw):
				edges.append(((j,i,z,1),(r,g,b,a)))
			
		i,j,r,g,b,a,z = i+di,j+dj,r+qr,g+qg,b+qb,a+qa,z+qz
	img.save(filename)
	return edges

def trif(points, col_interp):
	p0,p1,p2 = points[0],points[1],points[2]
	p0[0],p0[1],p0[2],p0[3]=p0[0]/p0[3],p0[1]/p0[3],p0[2]/p0[3],1
	p1[0],p1[1],p1[2],p1[3]=p1[0]/p1[3],p1[1]/p1[3],p1[2]/p1[3],1
	p2[0],p2[1],p2[2],p2[3]=p2[0]/p2[3],p2[1]/p2[3],p2[2]/p2[3],1
	
	p0=viewport_trans(p0)
	p1=viewport_trans(p1)
	p2=viewport_trans(p2)
	#passes the points along with corresponding colors
	l1 = (p0, p1, p0[2], p1[2])
	l2 = (p0, p2, p0[2], p2[2])
	l3 = (p1, p2, p1[2], p2[2])
	
	line1 = linec(l1, True, col_interp,False)
	line2 = linec(l2, True, col_interp,False)
	line3 = linec(l3, True, col_interp,False)
	coords = []

	for points1 in line1:
		coords.append(points1)
	for points2 in line2:
		if points2 not in coords:
			coords.append(points2)
	for points3 in line3:
		if points3 not in coords:
			coords.append(points3)
	coords = sorted(coords,key= lambda x: x[0][0])
	coords = sorted(coords,key= lambda x: x[0][1])

	#list of tuples of points with equal y values
	horiz = []
	for i in coords:
		for j in coords:
			if(i[0][1]==j[0][1] and j[0][0] != i[0][0] and (i,j) not in horiz and (j,i) not in horiz):
				horiz.append((i,j))

	filled = []
	for i in horiz:
		if (col_interp):
			a = [i[0][0][0],i[0][0][1],i[0][0][2],i[0][0][3],i[0][1]]
			b = [i[1][0][0],i[1][0][1],i[1][0][2],i[1][0][3],i[1][1]]
		else:
			a = [i[0][0][0],i[0][0][1],i[0][0][2],i[0][0][3],color_vals]
			b = [i[1][0][0],i[1][0][1],i[1][0][2],i[1][0][3],color_vals]
		
		linec((a,b),False, col_interp,True)

def viewport_trans(point):
	#scale x value:
	x,y = point[0],point[1]
	x,y = x+1, y+1
	return [round(x*width/2,5),round(y*height/2,5),point[2],point[3],point[4]]

def apply_matrix_trans(point):
	#vector for vertex:
	vec = [point[i] for i in range(4)]
	ans = [0 for i in range(4)]
	for i in range(4):
		for j in range(4):
			ans[i] += round(vec[j] * transformation[i][j],10)
	ans.append(point[4])
	return ans

def apply_proj_trans(point):
	vec = [point[i] for i in range(4)]
	ans = [0 for i in range(4)]
	for i in range(4):
		for j in range(4):
			ans[i] += round(vec[j] * projection[i][j],10)
	ans.append(point[4])
	return ans
 
def is_counter_clockwise(points):
	val = (points[1][1] - points[0][1]) * (points[2][0] - points[1][0]) - (points[1][0] - points[0][0]) * (points[2][1] - points[1][1])

	return True if val >0 else False

def cross_product(v1, v2):
	v3 = [v1[1]*v2[2] - v1[2]*v2[1],v1[2]*v2[0] - v1[0]*v2[2],v1[0]*v2[1] - v1[1]*v2[0]]
	return v3

def normalize(vec):
	vector = list(vec)
	mag = (round(vector[0]**2,5) + round(vector[1]**2,5) + round(vector[2]**2,5))**.5
	ans = [vector[0]/mag,vector[1]/mag,vector[2]/mag]
	return ans

def mult4by4(m1, m2):
	answer = [[0 for i in range(4)] for i in range(4)]
	for i in range(4):
		for j in range(4):
			answer[i][j]=0
			for t in range(4):
				answer[i][j] += m1[i][t]*m2[t][j]
	return answer

for line in lines:
	input(line)
