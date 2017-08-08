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


namespace
{
    using namespace GEO;
    vector<vec2> points;
    Mesh domain;
    Mesh rvd_mesh;
    Delaunay_var delaunay;
    RestrictedVoronoiDiagram_var rvd;

    

    void update_Delaunay()
    {
        delaunay->set_vertices(points.size(), &points.data()->x);
    }

    void create_random_points(index_t nb) {
        for(index_t i=0; i<nb; ++i) {
            points.push_back(
                    vec2(0.25, 0.25) + 
                    vec2(Numeric::random_float64()*0.5, Numeric::random_float64()*0.5)
                    );
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
        create_random_points(100);
        rvd->compute_RVD(rvd_mesh);
        std::cout<<"Vertices: " << rvd_mesh.vertices.nb() << std::endl;
        std::cout<<"Edges: " << rvd_mesh.edges.nb() << std::endl;
        std::cout<<"Facets: " << rvd_mesh.facets.nb() << std::endl;
        std::cout<<"Cells: " << rvd_mesh.cells.nb() << std::endl;
    }


    void display_rvd_mesh_facets()
    {
        glupSetColor3f(GLUP_FRONT_AND_BACK_COLOR, 0.0, 0.0, 0.0);
        glupSetMeshWidth(10);
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

//        index_t nf = rvd_mesh.facets.nb();
//        glupEnable(GLUP_VERTEX_COLORS);
//        glupBegin(GLUP_TRIANGLES);   
//        for(index_t i=0; i < nf; ++i)
//        {
//            index_t nv = rvd_mesh.facets.nb_vertices(i);
//            //glup_viewer_random_color_from_index(int(i));
//            for(index_t j=0; j < nv; ++j)
//            {
//                glupVertex(points[i]);
//                glupVertex(rvd_mesh.vertices.point(rvd_mesh.facets.vertex(i, j)));
//                glupVertex(rvd_mesh.vertices.point(rvd_mesh.facets.vertex(i, (j+1)%nv)));
//            }
//        }
//        glupEnd();
//        glupDisable(GLUP_VERTEX_COLORS);
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
       glupSetPointSize(GLfloat(20.0));
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
    //glup_viewer_set_overlay_func(overlay);
    //glup_viewer_set_mouse_func(mouse);
//    glup_viewer_add_key_func(
//        'k', Lloyd_relaxation, "One iteration of Lloyd relaxation"
//    );
//    glup_viewer_add_toggle(
//        'a', glup_viewer_is_enabled_ptr(GLUP_VIEWER_IDLE_REDRAW), "Animation"
//    );
    
    if(GEO::CmdLine::get_arg_bool("gfx:full_screen")) {
       glup_viewer_enable(GLUP_VIEWER_FULL_SCREEN);
    }

    glup_viewer_main_loop(argc, argv);

    return 0;
}
