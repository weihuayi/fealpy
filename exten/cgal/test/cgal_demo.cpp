
#include <geogram_gfx/glup_viewer/glup_viewer.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Random.h>
#include <CGAL/Bbox_2.h>
#include <list>


using namespace GEO;

class MeshAlg
{
public:
    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef CGAL::Triangulation_vertex_base_2<K> Vb;
    typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
    typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
    typedef CGAL::Exact_predicates_tag  Itag; 
    typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds, Itag> CDT;
    typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
    typedef CGAL::Delaunay_mesher_2<CDT, Criteria> Mesher;
    typedef CDT::Vertex_handle Vertex_handle;
    typedef K::Point_2 Point_2;
    typedef K::Segment_2 Segment_2;
    typedef CGAL::Bbox_2 Bbox_2;
    typedef CGAL::Random  Random;

    MeshAlg(double xmin=0.0, double ymin=0.0, double xmax=2.0, double ymax=1.0)
        : m_domain(xmin, ymin, xmax, ymax), m_cdt(), m_mesher(m_cdt) 
    {
        Vertex_handle va = m_cdt.insert(Point_2(xmin, ymin));
        Vertex_handle vb = m_cdt.insert(Point_2(xmax, ymin));
        Vertex_handle vc = m_cdt.insert(Point_2(xmax, ymax));
        Vertex_handle vd = m_cdt.insert(Point_2(xmin, ymax));
        m_cdt.insert_constraint(va, vb);
        m_cdt.insert_constraint(vb, vc);
        m_cdt.insert_constraint(vc, vd);
        m_cdt.insert_constraint(vd, va);

        generate_random_segments(5);
        insert_segment();

        m_mesher.set_criteria(Criteria(0.125, 0.1));
        m_mesher.refine_mesh();

    }
   

    /* 随机插入 n 个线段
     */
    void generate_random_segments(int n)
    {
        auto rand = CGAL::get_default_random();
        for(int i = 0; i < n; i++)
        {
            double x0 = rand.get_double(m_domain.xmin(), m_domain.xmax());
            double y0 = rand.get_double(m_domain.ymin(), m_domain.ymax());

            double x1 = rand.get_double(m_domain.xmin(), m_domain.xmax());
            double y1 = rand.get_double(m_domain.ymin(), m_domain.ymax());

            Point_2 p0(x0, y0);
            Point_2 p1(x1, y1);
            m_segs.push_back(Segment_2(p0, p1));
        }
    }
    
    void insert_segment()
    {
        for(auto it = m_segs.begin(); it != m_segs.end(); it++)
        {
            auto s = *it;
            auto v0 = m_cdt.insert(s[0]);
            auto v1 = m_cdt.insert(s[1]);
            m_cdt.insert_constraint(v0, v1);
        }
    }

    void draw_cdt_edges()
    {
        glupSetColor3f(GLUP_FRONT_AND_BACK_COLOR, 0, 0, 0);
        glupSetMeshWidth(3);
        glupBegin(GLUP_LINES);
        auto  it = m_cdt.finite_edges_begin();
        auto  end = m_cdt.finite_edges_end();
        for(; it != end; it++)
        {
            auto s = m_cdt.segment(it->first, it->second);
            glupVertex(vec2(s[0][0], s[0][1]));
            glupVertex(vec2(s[1][0], s[1][1]));
        }
        glupEnd();
    }

    void draw_cdt_triangles()
    {
        glupEnable(GLUP_VERTEX_COLORS);
        glupBegin(GLUP_TRIANGLES);
        auto it = m_cdt.finite_faces_begin();
        auto end = m_cdt.finite_faces_end();
        int i=0;
        for(; it != end; it++)
        {
            glup_viewer_random_color_from_index(i);
            for(int j = 0; j < 3; j++)
            {
                auto p = it->vertex(j)->point();
                glupVertex(vec2(p[0], p[1]));
            }
            i++;
        }
        glupEnd();
        glupDisable(GLUP_VERTEX_COLORS);  
    }

    void draw_segments()
    {
        glupSetColor3f(GLUP_FRONT_AND_BACK_COLOR, 1.0, 0.0, 0.0);
        glupSetMeshWidth(4);
        glupBegin(GLUP_LINES);
        for(auto it = m_segs.begin(); it != m_segs.end(); it++)
        {
            auto s = *it;
            glupVertex(vec2(s[0][0], s[0][1]));
            glupVertex(vec2(s[1][0], s[1][1]));
        }
        glupEnd();
    }

    void draw_points()
    {
        glupEnable(GLUP_LIGHTING);
        glupSetPointSize(GLfloat(8.0));
        glupDisable(GLUP_VERTEX_COLORS);
        glupSetColor3f(GLUP_FRONT_AND_BACK_COLOR, 0.0f, 1.0f, 1.0f);
        glupBegin(GLUP_POINTS);
        auto it = m_cdt.finite_vertices_begin();
        auto end = m_cdt.finite_vertices_end();
        for(; it != end; it++)
        {
            auto p = it->point();
            glupVertex(vec2(p[0], p[1]));
        }
        glupEnd();
        glupDisable(GLUP_LIGHTING);
    }


private:
    Bbox_2 m_domain;
    Criteria m_criteria;
    CDT m_cdt;
    Mesher m_mesher;
    std::list<Segment_2> m_segs;
};

class CGALApplication : public Application 
{
public:

    CGALApplication( int argc, char** argv,
            const std::string& usage
            ) : Application(argc, argv, usage) 
    {
        m_show_cdt = true;
        m_show_segment = true;
        m_show_points = true;
        glup_viewer_disable(GLUP_VIEWER_BACKGROUND);
        glup_viewer_disable(GLUP_VIEWER_3D);
    }

    static CGALApplication* instance() 
    {
        CGALApplication* result =
            dynamic_cast<CGALApplication*>(Application::instance());
        geo_assert(result != nil);
        return result;
    }

    virtual void draw_scene() 
    {
        glupSetColor3f(GLUP_FRONT_COLOR, 1.0f, 1.0f, 0.0f);
        glupSetColor3f(GLUP_BACK_COLOR, 1.0f, 0.0f, 1.0f);

        if(m_show_segment)
            m_alg.draw_segments();

        if(m_show_cdt)
        {
            m_alg.draw_cdt_edges();
            //m_alg.draw_cdt_triangles();
        }

        if(m_show_points)
            m_alg.draw_points();

    }

    virtual void init_graphics() 
    {
        Application::init_graphics();
    }
private:
    bool m_show_cdt;
    bool m_show_segment;
    bool m_show_points;
    MeshAlg  m_alg;
};

int main(int argc, char** argv) {
    CGALApplication app(argc, argv, "");
    app.start();
    return 0;
}
