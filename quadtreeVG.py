# lazy_quadtree_vg_plot.py
import math, time, heapq
from typing import List, Tuple, Dict, Optional
import matplotlib
matplotlib.use('TkAgg')
# ───────────────────────────── Geometry helpers ─────────────────────────────
Point   = Tuple[float, float]
Polygon = List[Point]
def orient(a, b, c):  # çapraz çarpım
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def on_seg(a, b, c):
    return min(a[0],c[0]) <= b[0] <= max(a[0],c[0]) and \
           min(a[1],c[1]) <= b[1] <= max(a[1],c[1])

def segments_intersect(p1, p2, q1, q2):
    o1,o2 = orient(p1,p2,q1), orient(p1,p2,q2)
    o3,o4 = orient(q1,q2,p1), orient(q1,q2,p2)
    if o1==o2==o3==o4==0:
        return (on_seg(p1,q1,p2) or on_seg(p1,q2,p2) or
                on_seg(q1,p1,q2) or on_seg(q1,p2,q2))
    return (o1*o2<0) and (o3*o4<0)

def point_in_convex(pt, poly):
    sign=None
    for i in range(len(poly)):
        cross=orient(poly[i],poly[(i+1)%len(poly)],pt)
        if cross!=0:
            cur=cross>0
            if sign is None: sign=cur
            elif sign!=cur:  return False
    return True

def segment_hits_poly(p,q,poly):
    if point_in_convex(((p[0]+q[0])/2,(p[1]+q[1])/2),poly):
        return True
    for i in range(len(poly)):
        if segments_intersect(p,q,poly[i],poly[(i+1)%len(poly)]):
            return True
    return False

# ───────────────────────────── Classic VG ───────────────────────────────────
class ClassicVisibilityGraph:
    def __init__(self,start:Point,goal:Point,obstacles:List[Polygon]):
        self.start,self.goal=start,goal
        self.obstacles=obstacles
        self.nodes:List[Point]=[]
        self.graph:List[List[Tuple[int,float]]]=[]
        self.time_ms:Optional[float]=None

    def _build_nodes(self):
        self.nodes=[self.start,self.goal]
        for p in self.obstacles:self.nodes.extend(p)

    def _build_edges(self):
        N=len(self.nodes)
        self.graph=[[] for _ in range(N)]
        for i in range(N):
            for j in range(i+1,N):
                p,q=self.nodes[i],self.nodes[j]
                if all(not segment_hits_poly(p,q,poly) for poly in self.obstacles):
                    d=math.hypot(q[0]-p[0],q[1]-p[1])
                    self.graph[i].append((j,d)); self.graph[j].append((i,d))

    def build(self):
        t=time.perf_counter()
        self._build_nodes(); self._build_edges()
        self.time_ms=(time.perf_counter()-t)*1000

    def shortest_path(self)->List[Point]:
        N,s,g=len(self.nodes),0,1
        INF=math.inf
        g_cost=[INF]*N; prev=[None]*N
        g_cost[s]=0
        h=lambda i: math.hypot(self.nodes[i][0]-self.goal[0],self.nodes[i][1]-self.goal[1])
        pq=[(h(s),s)]
        while pq:
            _,u=heapq.heappop(pq)
            if u==g:break
            for v,w in self.graph[u]:
                cand=g_cost[u]+w
                if cand<g_cost[v]:
                    g_cost[v]=cand; prev[v]=u
                    heapq.heappush(pq,(cand+h(v),v))
        if prev[g] is None: return []
        path=[]; cur=g
        while cur is not None: path.append(self.nodes[cur]); cur=prev[cur]
        return path[::-1]

# ─────────────────────── Quadtree + Lazy Edge VG ────────────────────────────
class QuadtreeVisibilityGraph(ClassicVisibilityGraph):
    class Quad:  # basit quadtree
        def __init__(self,bbox,depth): self.bbox=bbox; self.depth=depth; self.children=[]
        def leaf(self): return not self.children

    def __init__(self,start,goal,obstacles,env_bbox,max_depth=4):
        super().__init__(start,goal,obstacles)
        self.env_bbox=env_bbox; self.max_depth=max_depth
        self.cache:List[Dict[int,float]]=[]

    def _bbox_hits(self,bbox,poly):
        minx,miny,maxx,maxy=bbox; xs,ys=zip(*poly)
        return not(max(xs)<minx or min(xs)>maxx or max(ys)<miny or min(ys)>maxy)

    def _sub(self,bbox,depth):
        n=self.Quad(bbox,depth)
        if depth>=self.max_depth: return n
        if any(self._bbox_hits(bbox,p) for p in self.obstacles):
            minx,miny,maxx,maxy=bbox; mx,my=(minx+maxx)/2,(miny+maxy)/2
            for qb in [(minx,miny,mx,my),(mx,miny,maxx,my),
                       (minx,my,mx,maxy),(mx,my,maxx,maxy)]:
                n.children.append(self._sub(qb,depth+1))
        return n

    def _collect_centers(self,node,out):
        if node.leaf():
            minx,miny,maxx,maxy=node.bbox; cx,cy=(minx+maxx)/2,(miny+maxy)/2
            if any(point_in_convex((cx,cy),p) for p in self.obstacles): return
            if any(self._bbox_hits(node.bbox,p) for p in self.obstacles): out.append((cx,cy))
            return
        for ch in node.children:self._collect_centers(ch,out)

    def _build_nodes(self):
        super()._build_nodes()
        root=self._sub(self.env_bbox,0); centers=[]
        self._collect_centers(root,centers)
        self.nodes.extend(centers)
        self.cache=[{} for _ in range(len(self.nodes))]

    def _edge_cost(self,i,j):
        if j in self.cache[i]: return self.cache[i][j]
        p,q=self.nodes[i],self.nodes[j]
        seg_minx,seg_maxx=sorted([p[0],q[0]]); seg_miny,seg_maxy=sorted([p[1],q[1]])
        vis=True
        for poly in self.obstacles:
            xs,ys=zip(*poly)
            if seg_maxx<min(xs) or seg_minx>max(xs) or seg_maxy<min(ys) or seg_miny>max(ys): continue
            if segment_hits_poly(p,q,poly): vis=False; break
        w=math.inf if not vis else math.hypot(q[0]-p[0],q[1]-p[1])
        self.cache[i][j]=self.cache[j][i]=w; return w

    def shortest_path(self)->List[Point]:
        t0=time.perf_counter()
        N,s,g=len(self.nodes),0,1; INF=math.inf
        g_cost=[INF]*N; prev=[None]*N; g_cost[s]=0
        h=lambda i: math.hypot(self.nodes[i][0]-self.goal[0],self.nodes[i][1]-self.goal[1])
        pq=[(h(s),s)]
        while pq:
            _,u=heapq.heappop(pq)
            if u==g:break
            for v in range(N):
                if v==u: continue
                w=self._edge_cost(u,v)
                if w==math.inf: continue
                cand=g_cost[u]+w
                if cand<g_cost[v]:
                    g_cost[v]=cand; prev[v]=u
                    heapq.heappush(pq,(cand+h(v),v))
        self.time_ms=(time.perf_counter()-t0)*1000
        if prev[g] is None:return []
        path=[]; cur=g
        while cur is not None: path.append(self.nodes[cur]); cur=prev[cur]
        return path[::-1]

