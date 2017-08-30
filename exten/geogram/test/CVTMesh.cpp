#include <geogram_gfx/basic/GLSL.h>
#include <geogram_gfx/basic/GL.h>
#include <geogram_gfx/GLUP/GLUP.h>
#include <geogram_gfx/glup_viewer/glup_viewer.h>
#include <geogram/delaunay/delaunay.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/logger.h>
#include <geogram/numerics/predicates.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_io.h>

#include <geogram/voronoi/RVD.h>
#include <geogram/basic/geometry.h>

#include <iostream>

namespace
{
    using namespace GEO;
    vector<vec2> points;
    Mesh domain;
    Mesh domain_coarse_mesh;
    Mesh rvd_mesh;
    Delaunay_var delaunay;// Delaunay 对象的指针
    RestrictedVoronoiDiagram_var rvd; // rvd 的

    

    void update_Delaunay()
    {
        delaunay->set_vertices(points.size(), &points.data()->x);
    }

    void create_domain_coarse_mesh()
    {
        domain_coarse_mesh.clear(false, false);
        domain_coarse_mesh.vertices.set_dimension(3);
        domain_coarse_mesh.vertices.set_double_precision();

        index_t idx[] = {-1, -1, -1, -1};
        double *p = NULL;
        idx[0]  = domain_coarse_mesh.vertices.create_vertex();
        p = domain_coarse_mesh.vertices.point_ptr(idx[0]);
        p[0] = 0.0;
        p[1] = 0.0;
        p[2] = 0.0;

        idx[1] = domain_coarse_mesh.vertices.create_vertex();
        p = domain_coarse_mesh.vertices.point_ptr(idx[1]);
        p[0] = 1.0;
        p[1] = 0.0;
        p[2] = 0.0;

        idx[2] = domain_coarse_mesh.vertices.create_vertex();
        p = domain_coarse_mesh.vertices.point_ptr(idx[2]);
        p[0] = 1.0;
        p[1] = 1.0;
        p[2] = 0.0;

        idx[3] = domain_coarse_mesh.vertices.create_vertex();
        p = domain_coarse_mesh.vertices.point_ptr(idx[3]);
        p[0] = 0.0;
        p[1] = 1.0;
        p[2] = 0.0;

        domain_coarse_mesh.facets.create_triangle(idx[1], idx[2], idx[0]);
        domain_coarse_mesh.facets.create_triangle(idx[3], idx[0], idx[2]);
        return;
    }

    void create_random_points(index_t nb) 
    {
        create_domain_coarse_mesh();
        points.resize(nb);
        index_t nv = domain_coarse_mesh.vertices.nb();
        std::cout << "nv :" << nv << std::endl;
        vector<vec3> points3(nb-nv);
        for(index_t i = 0; i < nv; i++)
        {
            double * p = domain.vertices.point_ptr(i);
            points[i].x = p[0];
            points[i].y = p[1];
            std::cout<< points[i] << std::endl;
        }

        Delaunay_var dmesh = Delaunay::create(3,"BDEL2d");
        RestrictedVoronoiDiagram_var  local_rvd = RestrictedVoronoiDiagram::create(dmesh, &domain_coarse_mesh);
        local_rvd->compute_initial_sampling(&(points3.data())->x, nb-nv);

        for(index_t i = 0; i < nb - nv; i++)
        {
            points[i+nv].x = points3[i].x;
            points[i+nv].y = points3[i].y;
            std::cout << points[i+nv] << std::endl;
        }
        update_Delaunay();
    }

    void lloyd()
    {
        int nb = points.size();
        for(index_t i = domain.vertices.nb(); i < nb; i++)
        {
            vec2 g = points[i];
            std::cout << "g: " << g << std::endl;
            index_t nv = rvd_mesh.facets.nb_vertices(i);
            std::cout<< "nv:" << nv << std::endl;
            double area = 0.0;
            vec2 center(0.0, 0.0);
            for(index_t j=0; j < nv; j++)
            {
                vec3 p = rvd_mesh.vertices.point(rvd_mesh.facets.vertex(i, j));
                vec2 p0(p.x, p.y);
                std::cout << p0 << std::endl;
                p  = rvd_mesh.vertices.point(rvd_mesh.facets.vertex(i, (j+1)%nv));
                vec2 p1(p.x, p.y);
                std::cout << p1 << std::endl;
                std::cout << std::endl;

                vec2 c = Geom::barycenter(g, p0, p1);
                double a = Geom::triangle_area(g, p0, p1);
                center.x += c.x*a;
                center.y += c.y*a;
                area += a;
            }
            center.x /= area;
            center.y /= area;
            points[i].x = center.x;
            points[i].y = center.y;
        }
        update_Delaunay();
    }



    void create_domain()
    {
        domain.clear(false, false);
        domain.vertices.set_dimension(2);
        domain.vertices.set_double_precision();

        index_t idx[] = {-1, -1, -1, -1};
        double *p = NULL;
        idx[0]  = domain.vertices.create_vertex();
        p = domain.vertices.point_ptr(idx[0]);
        p[0] = 0.0;
        p[1] = 0.0;

        idx[1] = domain.vertices.create_vertex();
        p = domain.vertices.point_ptr(idx[1]);
        p[0] = 1.0;
        p[1] = 0.0;

        idx[2] = domain.vertices.create_vertex();
        p = domain.vertices.point_ptr(idx[2]);
        p[0] = 1.0;
        p[1] = 1.0;

        idx[3] = domain.vertices.create_vertex();
        p = domain.vertices.point_ptr(idx[3]);
        p[0] = 0.0;
        p[1] = 1.0;

        domain.facets.create_polygon(4, idx);
        return;
    }
    
    void init() {
        GEO::Graphics::initialize();
        
        glup_viewer_set_background_color(1.0, 1.0, 1.0);

        glup_viewer_add_toggle(
            'T', glup_viewer_is_enabled_ptr(GLUP_VIEWER_TWEAKBARS),
            "Toggle tweakbars"
        );

        glup_viewer_disable(GLUP_VIEWER_BACKGROUND);
        glup_viewer_disable(GLUP_VIEWER_3D);

        create_domain();
        delaunay = Delaunay::create(2,"BDEL2d");
        rvd = RestrictedVoronoiDiagram::create(delaunay, &domain);
        create_random_points(5);
        rvd->compute_RVD(rvd_mesh);

        for(int i = 0; i < 1; i++)
        {
            lloyd();
            rvd->compute_RVD(rvd_mesh);
        }

        std::cout<<"Vertices: " << rvd_mesh.vertices.nb() << std::endl;
        std::cout<<"Edges: " << rvd_mesh.edges.nb() << std::endl;
        std::cout<<"Facets: " << rvd_mesh.facets.nb() << std::endl;
        std::cout<<"Cells: " << rvd_mesh.cells.nb() << std::endl;
    }


    void display_rvd_mesh_facets()
    {
        glupSetColor3f(GLUP_FRONT_AND_BACK_COLOR, 0.0, 0.0, 0.0);
        glupSetMeshWidth(5);
        glupBegin(GLUP_LINES);
        for(index_t i=0; i< rvd_mesh.facets.nb(); ++i)
        {
            index_t nv = rvd_mesh.facets.nb_vertices(i);
            for(index_t j=0; j < nv; ++j)
            {
                glupVertex(rvd_mesh.vertices.point(rvd_mesh.facets.vertex(i, j)));
                glupVertex(rvd_mesh.vertices.point(rvd_mesh.facets.vertex(i, (j+1)%nv)));
            }
            
        }
        glupEnd();
    }

    /**
     * \brief Displays the border of the domain.
     */
    void display_domain() 
    {
        glupSetColor3f(GLUP_FRONT_AND_BACK_COLOR, 0.0, 0.0, 0.0);
        glupSetMeshWidth(4);
        glupBegin(GLUP_LINES);
        for(index_t i=0; i< domain.facets.nb(); ++i)
        {
            index_t nv = domain.facets.nb_vertices(i);
            for(index_t j=0; j < nv; ++j)
            {
                glupVertex(domain.vertices.point(domain.facets.vertex(i, j)));
                glupVertex(domain.vertices.point(domain.facets.vertex(i, (j+1)%nv)));
            }
            
        }
        glupEnd();
    }

    void display_points() 
    {
       glupEnable(GLUP_LIGHTING);
       glupSetPointSize(GLfloat(10.0));
       glupDisable(GLUP_VERTEX_COLORS);
       glupSetColor3f(GLUP_FRONT_AND_BACK_COLOR, 0.0f, 1.0f, 1.0f);
       glupBegin(GLUP_POINTS);
       for(index_t i=0; i<points.size(); ++i) {
           glupVertex(points[i]);
       }
       glupEnd();
       glupDisable(GLUP_LIGHTING);       
    }

    void display_Delaunay_triangles() 
    {
        glupSetColor3f(GLUP_FRONT_AND_BACK_COLOR, 0.7f, 0.7f, 0.7f);
        glupSetMeshWidth(1);
        glupBegin(GLUP_LINES);
        for(index_t c=0; c<delaunay->nb_cells(); ++c) {
            const signed_index_t* cell = delaunay->cell_to_v() + 3*c;
            for(index_t e=0; e<3; ++e) {
                signed_index_t v1 = cell[e];
                signed_index_t v2 = cell[(e+1)%3];
                glupVertex2dv(delaunay->vertex_ptr(index_t(v1)));
                glupVertex2dv(delaunay->vertex_ptr(index_t(v2)));
            }
        }
        glupEnd();
    }

    void display()
    {
        display_domain();
        display_points();
        display_rvd_mesh_facets();
        display_Delaunay_triangles();
    }

}

int main(int argc, char** argv)
{

    GEO::initialize();
    GEO::Logger::instance()->set_quiet(false);

    GEO::CmdLine::import_arg_group("standard");
    GEO::CmdLine::import_arg_group("algo");
    GEO::CmdLine::import_arg_group("gfx");    

    GEO::CmdLine::set_arg("sys:assert","abort");
    
    glup_viewer_set_region_of_interest(
        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f
    );

    glup_viewer_set_window_title(
        (char*) "Geogram 2d Domain test" 
    );

    glup_viewer_set_screen_size(1024,800);
    

    glup_viewer_set_init_func(init);
    glup_viewer_set_display_func(display);
    
    if(GEO::CmdLine::get_arg_bool("gfx:full_screen")) {
       glup_viewer_enable(GLUP_VIEWER_FULL_SCREEN);
    }

    glup_viewer_main_loop(argc, argv);

    return 0;
}
