
import polyscope as ps
import numpy as np 
from matplotlib import cm



def separate_faces_with_quantity(Faces_q,n_towers=10): 
    n_towers = int(np.amax(Faces_q[:,[3,4]])+1)
   
    Occupancy=np.zeros(n_towers)
    Dict={}
    Faces_separated=[]
    for face in Faces_q : 
        num1,num2 = face[[3,4]].astype(int)
        if num1!=-1:
            if Occupancy[num1]==0:
                Dict[num1]=[face[[0,1,2,5]]]
                Occupancy[num1]+=1
            else : 
                Dict[num1].append(face[[0,1,2,5]])
            
        if num2!=-1:
            if Occupancy[num2]==0:
                Dict[num2]=[face[[0,1,2,5]]]
                Occupancy[num2]+=1
            else : 
                Dict[num2].append(face[[0,1,2,5]])
            
    for i in sorted(list(Dict.keys())) : 
        Faces_separated.append(np.array(Dict[i]))
        
    return(np.array(Faces_separated))



##Support functions, simple mesh


def view_faces_values_on_embryo(Mesh,Vf,name_values = "Values",ps_mesh = None,colormap = cm.jet,min_to_zero=True,clean_before = True, clean_after=True,show=True,highlight_junctions=False,adapt_values = True, scattered = False):
    
    if scattered : 
        v,f,idx = Mesh.v_scattered,Mesh.f_scattered, Mesh.idx_scattered
    
    else : 
        v,f,idx = Mesh.v,Mesh.f, np.arange(len(Mesh.f))
        
    Values = Vf.copy()
    if adapt_values : 
        if min_to_zero : 
            Values-=np.amin(Values)
        Values/=(np.amax(Values)-np.amin(Values))

    Values = Values[idx]
    colors_face = colormap(Values)[:,:3]

    ps.init()
    
    if clean_before : 
        ps.remove_all_structures()


    if ps_mesh == None : 
        ps_mesh = ps.register_surface_mesh("Embryo", v,f[:,[0,1,2]])

    ps_mesh.set_color((0.3, 0.6, 0.8)) # rgb triple on [0,1]
    #ps_mesh.set_transparency(0.2)
    ps_mesh.add_color_quantity(name_values, colors_face, defined_on='faces',enabled=True)
    
    
    if highlight_junctions : 
        plot_trijunctions(Mesh,clean_before = False, clean_after = False, show=False, color = "uniform",value_color = np.ones(3))
    
    ps.set_ground_plane_mode("none") 
    
    if show : 
        ps.show()
    
    if clean_after : 
        ps.remove_all_structures()
        
    return(ps_mesh)
        

def view_vertex_values_on_embryo(Mesh,H,remove_trijunctions=True,ps_mesh = None,name_values = "Values",scattering_coeff = 1.0,clean_before = True, clean_after=True,show=True,highlight_junction = False,scattered = True):
    #H : Gaussian curvature at each vertex

    v,f = Mesh.v,Mesh.f

    Values = np.zeros(len(Mesh.f))

    Mesh.mark_trijunctional_vertices()
    valid_values = []
    indices_nan = []
    for i,face in enumerate(f[:,[0,1,2]]) : 
        a,b,c = face
        liste = []
        for vert_idx in face : 
            if remove_trijunctions : 
                if not Mesh.vertices[vert_idx].on_trijunction :
                    liste.append(H[vert_idx])
            else : 
                liste.append(H[vert_idx])
        if len(liste)>0 : 
            Values[i]=np.mean(np.array(liste))
            valid_values.append(Values[i])
        else : 
            indices_nan.append(i)

    mean = np.mean(np.array(valid_values))
    for i in indices_nan : 
        Values[i]=mean

    ps.init()

    Values-=np.amin(Values)
    Values/=np.amax(Values)

    if scattered :
        v,f,idx = Mesh.v_scattered,Mesh.f_scattered,Mesh.idx_scattered
        Values = Values[idx]

    colors_face = cm.jet(Values)[:,:3]

    if ps_mesh == None : 
        ps_mesh = ps.register_surface_mesh("Embryo", v,f[:,[0,1,2]])
    ps_mesh.add_color_quantity(name_values, colors_face, defined_on='faces',enabled=True)
       
    if highlight_junction : 
        plot_trijunctions(Mesh,clean_before = False, clean_after = False, show=False, color = "uniform",value_color = np.ones(3))

    ps.set_ground_plane_mode("none") 
    if show : 
        ps.show()
    
    if clean_after : 
        ps.remove_all_structures()
    
    return(ps_mesh)
    
def view_dict_values_on_mesh(Mesh,dict_values,alpha = 0.05,ps_mesh = None, clean_before = True, clean_after=True,show=True,scattered = False,name_values = "Values",alpha_values=True, min_value = None, max_value = None,cmap = cm.jet):

    v,f = Mesh.v,Mesh.f

    def find_values(face): 
        return(dict_values[tuple(face[[3,4]])])
    values = np.array(list(map(find_values,f)))
    values_values = np.array(list(dict_values.values()))
    if alpha_values:
        mint = np.quantile(values_values,alpha)
        maxt = np.quantile(values_values,1-alpha)
    else : 
        mint = min_value
        maxt = max_value
    values = values.clip(mint,maxt)
    print("Extremas of the "+name_values+" plotted : ",mint,maxt)
    values-=np.amin(mint)
    values/=np.amax(maxt-mint)
    
    ps_mesh = view_faces_values_on_embryo(Mesh,values,ps_mesh = ps_mesh,name_values = name_values,colormap = cmap,clean_before = clean_before, clean_after=clean_after,show=show,adapt_values=False,scattered= scattered)

    return(ps_mesh)
        
def plot_trijunctions(Mesh,Dict_trijunctional_values=None,clean_before = True, clean_after = True, show=True, color = "values",value_color = np.ones(3),cmap = cm.jet) : 
    #Dict_trijunctional_values=dict_line_tensions
    Dict_trijunctions = {}

    for edge in Mesh.half_edges : 
        if len(edge.twin)>1 :
            list_materials = []
            for a in edge.twin : 
                list_materials.append(Mesh.half_edges[a].incident_face.material_1)
                list_materials.append(Mesh.half_edges[a].incident_face.material_2)
            list_materials = np.unique(list_materials)
            key_junction = tuple(list_materials)
            Dict_trijunctions[key_junction] = Dict_trijunctions.get(key_junction,[]) + [[edge.origin.key,edge.destination.key]]

    ps.init()
    if clean_before : ps.remove_all_structures()

    if color == "uniform" : 
        plotted_edges = []
        edges = []
        verts = []
        i=0
        for key in Dict_trijunctions:
            if len(key)>=3: 
                for n in range(len(np.array(Dict_trijunctions[key]))): 
                    plotted_edges.append([2*i,2*i+1])
                    i+=1
                edges.append(np.array(Dict_trijunctions[key]))
                verts.append(Mesh.v[edges[-1]].reshape(-1,3))
        
        edges = np.vstack(edges)
        verts = np.vstack(verts)
        plotted_edges = np.array(plotted_edges)
        ps.register_curve_network("trijunctions", verts,plotted_edges,color = value_color)
        
    else : 
        
        edges = []
        verts = []
        plotted_edges=[]
        edges_values = []
        i=0
        for key in Dict_trijunctions : 
            if len(key)>=3:
                edges.append(np.array(Dict_trijunctions[key]))
                verts.append(Mesh.v[edges[-1]].reshape(-1,3))
                for n in range(len(edges[-1])): 
                    plotted_edges.append([2*i,2*i+1])
                    i+=1
                    edges_values.append(Dict_trijunctional_values.get(key,0))

        edges = np.vstack(edges)
        verts = np.vstack(verts)
        plotted_edges = np.array(plotted_edges)
        curv_net = ps.register_curve_network("trijunctions", verts,plotted_edges,color = np.random.rand(3))

        edges_values = np.array(edges_values)
        edges_values/= np.amax(edges_values)
        color_values = cmap(edges_values)[:,:3]
        curv_net.add_color_quantity("line tensions", color_values, defined_on='edges',enabled = True)



    if show : 
        ps.show()
        
    if clean_after : ps.remove_all_structures()






def plot_trijunctions_topo_change_viewer(Mesh,Dict_trijunctional_values=None,clean_before = True, clean_after = True, show=True, color = "values",value_color = np.ones(3)) : 

    Dict_trijunctions = {}

    for edge in Mesh.half_edges : 
        if len(edge.twin)>1 :
            list_materials = []
            for a in edge.twin : 
                list_materials.append(Mesh.half_edges[a].incident_face.material_1)
                list_materials.append(Mesh.half_edges[a].incident_face.material_2)
            list_materials = np.unique(list_materials)
            key_junction = tuple(list_materials)
            Dict_trijunctions[key_junction] = Dict_trijunctions.get(key_junction,[]) + [[edge.origin.key,edge.destination.key]]

    ps.init()
    if clean_before : ps.remove_all_structures()

    if color == "uniform" : 
        plotted_edges = []
        edges = []
        verts = []
        i=0
        for key in Dict_trijunctions:
            if len(key)>=3: 
                for n in range(len(np.array(Dict_trijunctions[key]))): 
                    plotted_edges.append([2*i,2*i+1])
                    i+=1
                edges.append(np.array(Dict_trijunctions[key]))
                verts.append(Mesh.v[edges[-1]].reshape(-1,3))
        
        edges = np.vstack(edges)
        verts = np.vstack(verts)
        plotted_edges = np.array(plotted_edges)
        ps.register_curve_network("trijunctions", verts,plotted_edges,color = value_color)
        
    else : 
        edges_r = []
        verts_r = []
        edges_g = []
        verts_g = []
        green_edges = []
        red_edges = []
        edges_values = []
        b=0
        r=0
        for key in Dict_trijunctions : 
            if len(key)>=3:
                edges_values.append(np.clip(Dict_trijunctional_values.get(key,0),0,None))
                if edges_values[-1]==1: 
                    for n in range(len(Dict_trijunctions[key])):
                        red_edges.append([2*r,2*r+1])
                        r+=1
                    edges_r.append(np.array(Dict_trijunctions[key]))
                    verts_r.append(Mesh.v[edges_r[-1]].reshape(-1,3))
                else : 
                    for n in range(len(Dict_trijunctions[key])):
                        green_edges.append([2*b,2*b+1])
                        b+=1


                    edges_g.append(np.array(Dict_trijunctions[key]))
                    verts_g.append(Mesh.v[edges_g[-1]].reshape(-1,3))


        verts_r = np.vstack(verts_r)
        verts_g = np.vstack(verts_g)
        red_edges = np.array(red_edges)
        green_edges = np.array(green_edges)
        curv_net = ps.register_curve_network("valid trijunctions", verts_g,green_edges,color = (0,1,0),transparency = 0.3)
        curv_net = ps.register_curve_network("bad trijunctions", verts_r,red_edges,color = (1,0,0),transparency = 1.0)


    if show : 
        ps.show()
        
    if clean_after : ps.remove_all_structures()














