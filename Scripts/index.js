var LABEL_COUNT = 5;
var DOCUMENT_COUNT = 500;
var Label_List = [];
var doc_topic_url, topic_key_word_url, label_url, topic_label_url, tsne_url;
var keywords;
var stock;
var summaries = [];
var topics = [];

(function(){
    $('#viz_modal').modal('hide')
    $('#label_modal').modal('hide')

    if(getQueryVariable("ds") == 1){
        doc_topic_url = './../Data/Anc_docTopic2.csv';
        topic_key_word_url = './../Data/topic_keywords1.csv';
        label_url = './../Data/Anc_label1.csv';
        topic_label_url = './../Data/Topic_Label1.csv';
        tsne_url = './../Data/tsne2.csv';
    }else{
        doc_topic_url = './../Data/Anc_docTopic2FoodReview.csv';
        topic_key_word_url = './../Data/topic_keywords2.csv';
        label_url = './../Data/Anc_label2.csv';
        topic_label_url = './../Data/Topic_Label2.csv';
        tsne_url = './../Data/tsne3.csv';
    }

    d3.csv(doc_topic_url, function(err, data){
        console.log(data)
        for (let index = 0; index < 10; index++) {
            topics.push(data[index]["Dominant_topic"])
        }
    })

    setTimeout(function(){console.log(topics);}, 1000)

    d3.csv(topic_label_url, function(err, data){
        LABEL_COUNT = data.length;
        data.forEach(element=>{
            Label_List.push(element.Label);
        })
    })

    d3.csv(label_url, function(err,data){
        data.forEach(element => {
            summaries.push(element.summary);
        })
    });

    console.log(summaries);

    keywords = {Topic1: "", Topic2: "", Topic3: "", Topic4: "", Topic5: "", Topic6: "", Topic7: "", Topic8: "", Topic9: "", Topic10: ""}
    d3.csv(topic_key_word_url, function(err, data){
        for (let mk = 0; mk < 3; mk++) {
            const element = data[mk];
            for (let ks = 1; ks < 11; ks++) {
                keywords["Topic"+ks] = keywords["Topic"+ks] + " "+element["Topic"+ks];
            }
        }
    });
    console.log("Keywords", keywords);

    console.log(LABEL_COUNT);

    $(function () {
        $('[data-toggle="tooltip"]').tooltip()
    })

    var div = d3.select("body").append("div")	
        .attr("class", "tooltip")				
        .style("opacity", 0);


    ///////////////////// TOPIC VIZ ///////////////////

    var topic_viz_svg = d3.select('#topic_viz'),
        to_width = $('#topic_viz').width(),
        to_height = $('#topic_viz').height();

    var q = d3_queue.queue(1)
        .defer(d3.csv, tsne_url)
        .defer(d3.csv, label_url)
        .awaitAll(topic_scatter_plot);
    
    // d3.csv('./../Data/tsne.csv', function(error,data){
    //     if (error) throw error;

    //     data.forEach(element => {
    //         element.X = +element.X
    //         element.Y = +element.Y
    //     });

    //     console.log(d3.extent(data, function(d) { return d.X; }))

    //     x.domain(d3.extent(data, function(d) { return d.X; }));
    //     y.domain(d3.extent(data, function(d) { return d.Y; }));

    //     // topic_viz_svg.append("g")
    //     //     .attr("class", "x axis")
    //     //     .attr("transform", "translate(-5," + to_height/2 +")")
    //     //     .call(xAxis)

    //     // topic_viz_svg.append("g")
    //     //     .attr("class", "y axis")
    //     //     .call(yAxis)
    //     //     .attr("transform", "translate("+to_width/2+",5)")

    //     topic_viz_svg.selectAll(".dot")
    //         .data(data)
    //       .enter().append("circle")
    //         .attr("class", "dot")
    //         .attr("r", function(d,i){if(i<500){return 3} else{return 10}})
    //         .style("opacity", function(d,i){if(i<500){return 0.8} else{return 0.5}})
    //         .attr("cx", function(d) { return x(d.X); })
    //         .attr("cy", function(d) { return y(d.Y); })
    //         .attr("stroke", function(d,i){if(i<500){return "black"}})
    //         .style("fill", function(d,i){if(i<500){return "grey"} else{return "blue"}});

        
    // });


    ////////////////////// TEXT LIST ////////////////////////////

    d3.csv(label_url, function(err, data){
        var ind = 0;
        new_data = [];
        for (let index = 0; index < 10; index++) {
            const element = data[index];
            if(element["Annotated Positive Document"] != '' || element["Correct Annotated Positive Document"] != ''){
                element.label = '<button type="button" class="btn btn-success btn-sm" style="float: left" role="button">'+ element["Label"] +'</button>';
            }else if(element["Annotated Negative Document"] != '' || element["Correct Annotated Negative Document"] != ''){
                element.label = '<button type="button" class="btn btn-danger btn-sm" style="float: left" role="button">'+ element["Label"] +'</button>';
            }else {
                element.label = '';
            }
            new_data.push(element);
        }
        new_data.forEach(element => {
            console.log(JSON.stringify(element))
            var name = element.Document
            $('#text_list_inside').append('<div class="card" style="width: 100%"><div class="card-body" style="padding: 10px">'+
                '<h5 class="card-title"><a class="btn btn-link btn-sm text-dark" style="text-decoration: none" data-toggle="collapse" href="#collapseExample'+ ind +'" role="button" aria-expanded="false" aria-controls="collapseExample'+ ind +'">'+
                '<b>'+ element.Document + ': '+ topics[ind]+'</b></a></h5><div class="doc_labels" style="width: 100%">'+
                element.label +
                '<i class="material-icons" onClick="show_doc_modal(\''+ name +'\','+ element.id +',\''+ summaries[ind].replace(/[^a-z0-9 \.]+/g, "") +'\')" style="cursor: pointer; float: left; margin-top: 2px">add</i></div><div class="collapse mt-2" id="collapseExample'+ ind +'"><div class="card card-body">'+ element.summary +'</div></div></div></div>')
            ind++;
        });
    });

    // d3.csv('./../Data/doc_topic.csv', function(error,data){
    //     var ind = 0;
    //     new_data = []
    //     for (let index = 0; index < 10; index++) {
    //         const element = data[index];
    //         new_data.push(element);
    //     }
    //     new_data.forEach(element => {
    //         console.log(JSON.stringify(element))
    //         var name = element.Document
    //         $('#text_list_inside').append('<div class="card" style="width: 100%"><div class="card-body" style="padding: 10px">'+
    //             '<h5 class="card-title"><a class="btn btn-link btn-sm text-dark" style="text-decoration: none" data-toggle="collapse" href="#collapseExample'+ ind +'" role="button" aria-expanded="false" aria-controls="collapseExample'+ ind +'">'+
    //             '<b>'+ element.Document +'</b></a></h5><div class="doc_labels" style="width: 100%">'+
    //             '<button type="button" class="btn btn-danger btn-sm" style="float: left" role="button">Label 1</button>'+
    //             '<i class="material-icons" onClick="show_doc_modal(\''+ name +'\')" style="cursor: pointer; float: left; margin-top: 2px">add</i></div><div class="collapse mt-2" id="collapseExample'+ ind +'"><div class="card card-body">Anim pariatur cliche reprehenderit, enim eiusmod high life accusamus terry richardson ad squid. Nihil anim keffiyeh helvetica, craft beer labore wes anderson cred nesciunt sapiente ea proident.</div></div></div></div>')
    //         ind++;
    //     });
    // })

    ////////////////// WORD CLOUD ///////////////////

    draw_cloud(1);

    ////////////////// DENDOGRAM ///////////////////////////
    
    
    d3.csv(doc_topic_url, function(error,data){
        var dendo_svg = d3.select('#dendo_viz'),
            dendo_width = $('#dendo_viz').width(),
            dendo_height = $('#dendo_viz').height(),
            nodeRadius = 4.5,
            radius = to_width*3/4;
            console.log(dendo_height);
        
        var mainGroup = dendo_svg.append('g')
            .attr("transform", "translate(" + radius + "," + radius + ")");

        new_data = {name: "root", children: []}
        for (let i = 1; i <= 10; i++) {
            children1 = {name: "Topic "+i, children: []}
            new_child = data.filter(function(d){return d.Dominant_topic == String("Topic"+i);})
            new_child.forEach(element => {
                children2 = {name: element.Document, children: []}
                children1.children.push(children2)
            });
            new_data.children.push(children1)
        }
        console.log(new_data)

        var cluster = d3.cluster()
            .size([360, radius - 50]);

        var root = d3.hierarchy(new_data, function(d) {
            return d.children;
        });

        cluster(root);

        var linksGenerator = d3.linkRadial()
            .angle(function(d) { return d.x / 180 * Math.PI; })
            .radius(function(d) { return d.y; });

        mainGroup.selectAll('path')
            .data(root.links())
            .enter()
            .append('path')
            .attr("d", linksGenerator)
            .attr("fill", "none")
            .attr("stroke", "#ccc")
        

        var nodeGroups = mainGroup.selectAll("g")
            .data(root.descendants())
            .enter()
            .append("g")
            .attr("transform", function(d) {
                return "rotate(" + (d.x - 90) + ")translate(" + d.y + ")";
            });

        nodeGroups.append("circle")
            .attr("r", nodeRadius)
            .attr("fill", "none")
            .attr("stroke", "#ccc")
            .attr('stroke-width', 3)
            .on('mouseover', function(d, i){
                if(d.height == 0){
                    div.transition()
                        .duration(200)
                        .style("opacity", .9);
                    div.html(d.data.name)
                        .style("left", (d3.event.pageX) + "px")
                        .style("top", (d3.event.pageY - 28) + "px");
                }
                d3.select(this)
                    .attr('stroke', 'purple')
                    .attr('stroke-width', 3)
            }).on('mouseout', function(d, i){
                if(d.height == 0){
                    div.transition()
                        .duration(200)
                        .style("opacity", 0);
                }
                d3.select(this)
                    .attr('stroke', '#ccc')
                    .attr('stroke-width', 2)
            }).on('click', function(d,i){
                console.log(i)
                if(d.height == 1){
                    draw_cloud(i)
                }
            });

        nodeGroups.append("text")
            .attr("dy", ".31em")
            .attr("x", function(d) { return d.x < 180 === !d.children ? 6 : -6; })
            .style("text-anchor", function(d) { return d.x < 180 === !d.children ? "start" : "end"; })
            .attr("transform", function(d) { return "rotate(" + (d.x < 180 ? d.x - 90 : d.x + 90) + ")"; })
            .text(function(d) { if(d.height > 0){return d.data.name;} });

    })

    //////////////// Modal Viz //////////////////
    
    d3.csv(doc_topic_url, function(error,data){
        var modal_svg = d3.select('#modal_svg'),
            dendo_width = $(window).width()-50,
            dendo_height = $(window).height()-50,
            nodeRadius = 4.5,
            radius = dendo_width/2;

        var mainGroup = modal_svg.append('g')
            .attr("transform", "translate(" + radius + "," + radius + ")");

        new_data = {name: "root", children: []}
        for (let i = 1; i <= 10; i++) {
            children1 = {name: "Topic "+i, children: []}
            new_child = data.filter(function(d){return d.Dominant_topic == String("Topic"+i);})
            new_child.forEach(element => {
                children2 = {name: element.Document, children: []}
                children1.children.push(children2)
            });
            new_data.children.push(children1)
        }
        console.log(new_data)

        var cluster = d3.cluster()
            .size([360, radius - 50]);

        var root = d3.hierarchy(new_data, function(d) {
            return d.children;
        });

        cluster(root);

        var linksGenerator = d3.linkRadial()
            .angle(function(d) { return d.x / 180 * Math.PI; })
            .radius(function(d) { return d.y; });

        mainGroup.selectAll('path')
            .data(root.links())
            .enter()
            .append('path')
            .attr("d", linksGenerator)
            .attr("fill", "none")
            .attr("stroke", "#ccc")
        

        var nodeGroups = mainGroup.selectAll("g")
            .data(root.descendants())
            .enter()
            .append("g")
            .attr("transform", function(d) {
                return "rotate(" + (d.x - 90) + ")translate(" + d.y + ")";
            });

        nodeGroups.append("circle")
            .attr("r", nodeRadius)
            .attr("fill", "none")
            .attr("stroke", "#ccc")
            .attr('stroke-width', 2)
            .style('cursor', 'pointer')
            .on('mouseover', function(d, i){
                if(d.height == 0){
                    div.transition()
                        .duration(200)
                        .style("opacity", .9);
                    div.html(d.data.name)
                        .style("left", (d3.event.pageX) + "px")
                        .style("top", (d3.event.pageY - 28) + "px");
                }
                d3.select(this)
                    .attr('stroke', 'purple')
                    .attr('stroke-width', 3)
            }).on('mouseout', function(d, i){
                if(d.height == 0){
                    div.transition()
                        .duration(200)
                        .style("opacity", 0);
                }
                d3.select(this)
                    .attr('stroke', '#ccc')
                    .attr('stroke-width', 2)
            })

        nodeGroups.append("text")
            .attr("dy", ".31em")
            .attr("x", function(d) { return d.x < 180 === !d.children ? 6 : -6; })
            .style("text-anchor", function(d) { return d.x < 180 === !d.children ? "start" : "end"; })
            .attr("transform", function(d) { return "rotate(" + (d.x < 180 ? d.x - 90 : d.x + 90) + ")"; })
            .text(function(d) { if(d.height > 0){return d.data.name;} });

    });

    /////////////////////////// LABEL LIST //////////////////////////

    d3.csv(label_url, function(error, data){
        var final_data = []
        for(let i=0;i<LABEL_COUNT;i++){
            var obj = {label_name: Label_List[i], c_p_a: 0, p_a: 0, c_n_a: 0, n_a:0};
            final_data.push(obj);
        }
        for(let i=0;i<LABEL_COUNT;i++){
            tem = data.filter(function(d){return d['Correct Annotated Positive Document'] === Label_List[i]});
            final_data[i].c_p_a = tem.length;
            tem = data.filter(function(d){return d['Annotated Positive Document'] === Label_List[i]});
            final_data[i].p_a = tem.length;
            tem = data.filter(function(d){return d['Correct Annotated Negative Document'] === Label_List[i]});
            final_data[i].c_n_a = tem.length;
            tem = data.filter(function(d){return d['Annotated Negative Document'] === Label_List[i]});
            final_data[i].n_a = tem.length;
        }
        var l_l_svg = d3.select('#l_l_svg'),
            l_l_width = $('#l_l_svg').width(),
            l_l_height = $('#l_l_svg').height();

        var x = d3.scaleLinear().domain([0,100]).range([0, (l_l_width)/2]);
        var marg = 45;
        for (let index = 0; index < LABEL_COUNT; index++) {
            var new_g = l_l_svg.append("g");

            var title = new_g.append("text")
                .text(Label_List[index])
                .attr("transform", "translate("+ l_l_width/6+ "," + (marg+25) + ")")
                .style("font-family", "Lato, sans-serif")

            var bg_rect = new_g.append("rect")
                .attr("height", 25)
                .attr("width", x(200))
                .attr("transform", "translate("+ l_l_width/3+ "," + marg + ")")
                .attr("fill","#a7d069")

            var bg_rect = new_g.append("rect")
                .attr("height", 25)
                .attr("width", x(200))
                .attr("transform", "translate("+ l_l_width/3+ "," + (marg+26) + ")")
                .attr("fill","#f1b05c")
            
            var c_p = new_g.append("rect")
                .attr("height", 10)
                .attr("width", x(final_data[index].c_p_a))
                .attr("transform", "translate("+ l_l_width/3+ "," + (marg+5) + ")")
                .attr("fill","green")

            var p = new_g.append("rect")
                .attr("height", 10)
                .attr("width", x(final_data[index].p_a))
                .attr("transform", "translate("+ l_l_width/3+ "," + (marg+15) + ")")
                .attr("fill","green")

            var c_n = new_g.append("rect")
                .attr("height", 10)
                .attr("width", x(final_data[index].c_n_a))
                .attr("transform", "translate("+ l_l_width/3+ "," + (marg+26) + ")")
                .attr("fill","red")

            var n = new_g.append("rect")
                .attr("height", 10)
                .attr("width", x(final_data[index].n_a))
                .attr("transform", "translate("+ l_l_width/3+ "," + (marg+36) + ")")
                .attr("fill","red")
            
            marg+=65;
        }
    })

    ////////////// TOPIC WEIGHT /////////////////////

    d3.csv(topic_label_url, function(error, data){
        var t_w_svg = d3.select('#t_w_svg'),
            t_w_width = $('#t_w_svg').width(),
            t_w_height = $('#t_w_svg').height();
        console.log(data)
        var r = d3.scaleSqrt().domain([0,1]).range([0,20])

        var x_ax_g = t_w_svg.append("g");
        var x_la = Label_List;
        x_ax_g.selectAll("text")
            .data(x_la)
            .enter().append("text")
            .text(function(d){return d;})
            .attr("transform", "translate(80, 80)")
            .attr("y", function(d,i){return 45*i+10})

        var y_la = ["Topic1", "Topic2", "Topic3", "Topic4", "Topic5", "Topic6", "Topic7", "Topic8", "Topic9", "Topic10"];
        var y_ax_g = t_w_svg.append("g");
        y_ax_g.selectAll("text")
            .data(y_la)
            .enter().append("text")
            .text(function(d){return d;})
            .attr("transform", "translate(140,60)")
            .attr("x", function(d,i){return (50*i)+10})
            

        var marg = 0;
        for (let index = 0; index < LABEL_COUNT; index++) {
            da = []
            for(ik = 1;ik<=10;ik++){
                da.push({["Topic"+ik]: data[index]["Topic"+ik]});
            }
            console.log(da)
            var new_g = t_w_svg.append("g");
            new_g.selectAll("circle")
                .data(da)
                .enter().append("circle")
                .attr("r", function(d,i){console.log("Topic"+i + " - "+JSON.stringify(d)); return r(d["Topic"+(i+1)]);})
                .attr("cx", function(d,i){return (i*50)+5;})
                .attr("cy", marg+6)
                .attr("fill", "red")
                .attr("transform", "translate(160, 80)")

            marg+=45;
        }
    })

    ///////////////////// TREEMAP /////////////////////////////

    d3.csv(doc_topic_url, function(error,data){
        var treem_svg = d3.select('#treem_viz'),
            treem_width = $('#topic_viz').width(),
            treem_height = $('#topic_viz').height(),
            formatNumber = d3.format(","),
            transitioning;
        var margin = {top: 30, right: 0, bottom: 20, left: 0};

        var x = d3.scaleLinear()
            .domain([0, treem_width])
            .range([0, treem_width]);
        var y = d3.scaleLinear()
            .domain([0, treem_height])
            .range([0, treem_height]);

        var treemap = d3.treemap()
            .size([treem_width, treem_height])
            .paddingInner(0)
            .round(false);

        var tr_g = treem_svg.append("g")
            .attr("transform", "translate(15,15)")
            .style("shape-rendering", "crispEdges");

        var grandparent = tr_g.append("g")
            .attr("class", "grandparent");
            grandparent.append("rect")
                .attr("y", 0)
                .attr("width", treem_width)
                .attr("height", margin.top)
                .attr("fill", '#bbbbbb');
            grandparent.append("text")
                .attr("x", 6)
                .attr("y", 5)
                .attr("dy", ".75em");
        
        
        new_data = {name: "root", children: []}
        for (let i = 1; i <= 10; i++) {
            children1 = {name: "Topic "+i, rate: 0.03, children: [],value: 0.5}
            new_child = data.filter(function(d){return d.Dominant_topic == String("Topic"+i);})
            new_child.forEach(element => {
                children2 = {name: element.Document, rate: 0.03, value: 0.5}
                children1.children.push(children2)
            });
            new_data.children.push(children1)
        }
        console.log(new_data)

        var root = d3.hierarchy(new_data)
        console.log(root);
        treemap(root
            .sum(function (d) {
                console.log(d.value)
                return d.value;
            })
            .sort(function (a, b) {
                return b.height - a.height;
            })
        );
        display(root);

        function display(d) {
            // write text into grandparent
            // and activate click's handler
            grandparent
                .datum(d.parent)
                .on("click", transition)
                .select("text")
                .text(name(d));
            // grandparent color
            grandparent
                .datum(d.parent)
                .select("rect")
                .attr("fill", function () {
                    return '#bbbbbb'
                });
            var g1 =tr_g.insert("g", ".grandparent")
                .datum(d)
                .attr("class", "depth")
                .attr("transform","translate(0,30)");
            var g = g1.selectAll("g")
                .data(d.children)
                .enter().
                append("g");
            // add class and click handler to all g's with children
            g.filter(function (d) {
                return d.children;
            })
                .classed("children", true)
                .on("click", transition);
            g.selectAll(".child")
                .data(function (d) {
                    return d.children || [d];
                })
                .enter().append("rect")
                .attr("class", "child")
                .call(rect);
            // add title to parents
            g.append("rect")
                .attr("class", "parent")
                .call(rect)
                .append("title")
                .text(function (d){
                    return d.data.name;
                });
            /* Adding a foreign object instead of a text object, allows for text wrapping */
            g.append("foreignObject")
                .call(rect)
                .attr("class", "foreignobj")
                .append("xhtml:div")
                .attr("dy", ".75em")
                .html(function (d) {
                    return '' +
                        '<p class="title"> ' + d.data.name + '</p>'
                    ;
                })
                .attr("class", "textdiv"); //textdiv class allows us to style the text easily with CSS
            function transition(d) {
                if (transitioning || !d) return;
                transitioning = true;
                var g2 = display(d),
                    t1 = g1.transition().duration(650),
                    t2 = g2.transition().duration(650);
                // Update the domain only after entering new elements.
                x.domain([d.x0, d.x1]);
                y.domain([d.y0, d.y1]);
                // Enable anti-aliasing during the transition.
               tr_g.style("shape-rendering", null);
                // Draw child nodes on top of parent nodes.
               tr_g.selectAll(".depth").sort(function (a, b) {
                    return a.depth - b.depth;
                });
                // Fade-in entering text.
                g2.selectAll("text").style("fill-opacity", 0);
                g2.selectAll("foreignObject div").style("display", "none");
                /*added*/
                // Transition to the new view.
                t1.selectAll("text").call(text).style("fill-opacity", 0);
                t2.selectAll("text").call(text).style("fill-opacity", 1);
                t1.selectAll("rect").call(rect);
                t2.selectAll("rect").call(rect);
                /* Foreign object */
                t1.selectAll(".textdiv").style("display", "none");
                /* added */
                t1.selectAll(".foreignobj").call(foreign);
                /* added */
                t2.selectAll(".textdiv").style("display", "block");
                /* added */
                t2.selectAll(".foreignobj").call(foreign);
                /* added */
                // Remove the old node when the transition is finished.
                t1.on("end.remove", function(){
                    this.remove();
                    transitioning = false;
                });
            }
            return g;
        }
        function text(text) {
            text.attr("x", function (d) {
                return x(d.x) + 6;
            })
                .attr("y", function (d) {
                    return y(d.y) + 6;
                });
        }
        function rect(rect) {
            rect
                .attr("x", function (d) {
                    return x(d.x0);
                })
                .attr("y", function (d) {
                    return y(d.y0);
                })
                .attr("width", function (d) {
                    return x(d.x1) - x(d.x0);
                })
                .attr("height", function (d) {
                    return y(d.y1) - y(d.y0);
                })
                .attr("fill", function (d) {
                    return '#bbbbbb';
                });
        }
        function foreign(foreign) { /* added */
            foreign
                .attr("x", function (d) {
                    return x(d.x0);
                })
                .attr("y", function (d) {
                    return y(d.y0);
                })
                .attr("width", function (d) {
                    return x(d.x1) - x(d.x0);
                })
                .attr("height", function (d) {
                    return y(d.y1) - y(d.y0);
                });
        }
        function name(d) {
            return breadcrumbs(d) +
                (d.parent
                ? " -  Click to zoom out"
                : " - Click inside square to zoom in");
        }
        function breadcrumbs(d) {
            var res = "";
            var sep = " > ";
            d.ancestors().reverse().forEach(function(i){
                res += i.data.name + sep;
            });
            return res
                .split(sep)
                .filter(function(i){
                    return i!== "";
                })
                .join(sep);
        }

    })

})()

function show_modal(){
    $('#viz_modal').modal('show');
}

docu_name = ''

function show_doc_modal(document, id, summary){
    // element = JSON.parse(element)
    // console.log(element)
    $('#label_modal').modal('show')
    docu_name = document;
    $('#exampleModalLongTitle').text(document)
    $('#new_label_in').html('<div>'+ summary +'</div><input type="text" id="new_label" class="form-control" placeholder="New Label Name" aria-label="New Label Name" aria-describedby="basic-addon2">'+
        '<div class="input-group-append">'+
        '<button class="btn btn-outline-secondary" type="button" onclick="add_new_label('+ id +')">Add Label</button>'+
        '</div>');
    var q = d3_queue.queue(1)
        .defer(d3.csv, topic_label_url)
        .defer(d3.csv, label_url)
        .awaitAll(draw);

}

function add_new_label(id){
    console.log($('#new_label').val(), id)
    $.ajax({
        url: 'http://127.0.0.1:8081/?id='+id+'&posno=yes&ds='+ getQueryVariable('ds') +'&newlabel=yes&label_id='+$('#new_label').val(),
        type: 'GET',
        success: function(res){
            if(res == 'success'){
                window.location.href = window.location.href;
            }
        }
    })
    setTimeout(function(){ console.log("here"); location.reload(true) }, 3000);
}

function draw(err, data){
    $('#la_lis').html('');
    var label_data = data[0];
    var doc_data = data[1];

    var doc_row = doc_data.filter(function(d){var doc_s = String(d.Document); console.log(docu_name.replace(/(\r\n|\n|\r)/gm, ""), doc_s.replace(/(\r\n|\n|\r)/gm, "")); return doc_s === docu_name.replace(/(\r\n|\n|\r)/gm, "");})
    console.log(doc_row)
    // if(do)
    var cu_label_id = doc_row[0].Label.substring(5,6);
    console.log(cu_label_id)
    var lab_stat1 = 'btn-light';
    var lab_stat2 = 'btn-light';
    var doc_no = doc_row[0]['id'];
    if(doc_row[0]['Annotated Positive Document'] === doc_row[0].Label){
        lab_stat1 = 'btn-success';
    }else if(doc_row[0]['Correct Annotated Positive Document'] === doc_row[0].Label){
        lab_stat1 = 'btn-success';
    }else if(doc_row[0]['Annotated Negative Document'] === doc_row[0].Label){
        lab_stat2 = 'btn-danger';
    }else if(doc_row[0]['Correct Annotated Negative Document'] === doc_row[0].Label){
        lab_stat2 = 'btn-danger';
    }
    var ind = 0
    label_data.forEach(element => {
        if(element.Label != doc_row[0].Label){
            $('#la_lis').append('<div class="label_title">'+ label_data[ind]['Label'] +'</div>'+
                '<div class="btn-group" role="group" aria-label="Basic example">'+
                '<button type="button" onclick="upGML(\''+element.Label+'\',\'yes\','+ doc_no +', \'btn-light\')" class="btn btn-light">Yes</button>'+
                '<button type="button" onclick="upGML(\''+element.Label+'\',\'no\','+ doc_no +', \'btn-light\')" class="btn btn-light">No</button>'+
                '</div>'
                )    
        }else{
            $('#la_lis').append('<div class="label_title">'+ label_data[ind]['Label'] +'</div>'+
                '<div class="btn-group" role="group" aria-label="Basic example">'+
                '<button type="button" onclick="upGML(\''+element.Label+'\',\'yes\','+ doc_no +', \''+ lab_stat1 +'\')" class="btn '+ lab_stat1 +'">Yes</button>'+
                '<button type="button" onclick="upGML(\''+element.Label+'\',\'no\','+ doc_no +', \''+ lab_stat2 +'\')" class="btn '+ lab_stat2 +'">No</button>'+
                '</div>'
                )
        }
        ind++;
    })
}

function topic_scatter_plot(err, data){
    if (err) throw error;

    var div = d3.select("body").append("div")	
        .attr("class", "tooltip")				
        .style("opacity", 0);

    var topic_viz_svg = d3.select("#topic_viz"),
        to_width = $('#topic_viz').width(),
        to_height = $('#topic_viz').height();
    var index = 0;
    var topic_index = 1;
    var scatter_plot_x = d3.scaleLinear()
        .range([0, to_width]);

    var scatter_plot_y = d3.scaleLinear()
        .range([to_height, 0]);

    var xAxis = d3.axisBottom(scatter_plot_x);

    var yAxis = d3.axisLeft(scatter_plot_y);
    console.log(data[1][index]["Annotated Positive Document"]);
    data[0].forEach(element => {
        element.X = +element.X;
        element.Y = +element.Y;
        if(index < 500){
            element.name = data[1][index].Document;
            if(data[1][index]["Annotated Positive Document"] != '' || data[1][index]["Correct Annotated Positive Document"] != ''){
                element.label_type = "green";
            }else if(data[1][index]["Annotated Negative Document"] != '' || data[1][index]["Correct Annotated Negative Document"] != ''){
                element.label_type = "red";
            }else {
                element.label_type = "grey";
            }
        }else if(index >= 500){
            element.name = "Topic "+topic_index + ": "+ keywords["Topic"+(index-499)];
            topic_index++;
        }
        index++;
    });

    console.log(data[0]);

    scatter_plot_x.domain(d3.extent(data[0], function(d) { return d.X; }));
    scatter_plot_y.domain(d3.extent(data[0], function(d) { return d.Y; }));
    console.log(scatter_plot_y(data[0][0]["Y"]));
    topic_viz_svg.selectAll(".dot")
        .data(data[0])
        .enter().append("circle")
        .attr("class", "dot")
        .attr("r", function(d,i){if(i<500){return 3} else{return 10}})
        .style("opacity", function(d,i){if(i<500){return 0.8} else{return 0.5}})
        .attr("cx", function(d) { return scatter_plot_x(d.X); })
        .attr("cy", function(d) { return scatter_plot_y(d.Y); })
        .attr("stroke", function(d,i){if(i<500){return "black"}})
        .style("fill", function(d,i){if(i<500){return d["label_type"]} else{return "blue"}})
        .style("cursor", 'pointer')
        .on("mouseover", function(d,i){
            div.transition()
                .duration(200)
                .style("opacity", .9);
            div.html(d["name"])
                .style("left", (d3.event.pageX) + "px")
                .style("top", (d3.event.pageY - 28) + "px");
        }).on("mouseout", function(d,i){
            div.transition()
                .duration(200)
                .style("opacity", 0);
        }).on("click", function(d,i){
            if(i>=500){
                draw_cloud(i-DOCUMENT_COUNT+1);
            }
        })
}

function draw_cloud(index){

    var w_c_svg = d3.select('#w_c_inside'),
        wc_width = $('#w_c_inside').width(),
        wc_height = $('#w_c_inside').height();

    d3.selectAll("#w_c_inside > *").remove();

    var g = w_c_svg.append("g")
            .attr("transform", "translate(0,10)")

    d3.csv(topic_key_word_url, function(error,data){
        if (error) throw error;
        
        d_data = [];
        // data = data[index]
        
        // console.log(data);
        // data.sort(function(x, y){
        //     return d3.descending(x.index, y.index);
        // })
        // for(let i=1;i<=30;i++){
        //     d_data.push({WordL: data["R"+i]})
        //     console.log(data["R"+i]);
            
        // }

        console.log(d_data)
        

        var fontSize = d3.scalePow().exponent(5).domain([0,1]).range([12,40]);

        var layout = d3.layout.cloud()
            .size([wc_width, wc_height])
            .timeInterval(20)
            .words(data)
            .rotate(function(d) { return 0; })
            .fontSize(function(d,i) { return fontSize(Math.random()); })
            .fontWeight(["bold"])
            .text(function(d) { return d['Topic'+(index)]; })
            .spiral("rectangular") // "archimedean" or "rectangular"
            .on("end", draw)
            .start();

        var wordcloud = g.append("g")
            .attr('class','wordcloud')
            .attr("transform", "translate(" + wc_width/2 + "," + wc_height/2 + ")");

        g.append("g")
            .attr("class", "axis")
            .attr("transform", "translate(0," + wc_height + ")")
            .selectAll('text')
            .style('font-size','20px')
            .style('font','Lato, sans-serif');

        function draw(words) {
            wordcloud.selectAll("text")
                .data(words)
                .enter().append("text")
                .attr('class','word')
                .style("font-size", function(d) { return d.size + "px"; })
                .style("font-family", function(d) { return d.font; })
                .style("font-family", "Lato, sans-serif")
                //.style("fill", function(d) { 
                    //var paringObject = data.filter(function(obj) { return obj.Team_CN === d.text});
                    // return color(paringObject[0].category); 
                //})
                .attr("text-anchor", "middle")
                .attr("transform", function(d) { return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")"; })
                .text(function(d) { return d.text; });
        };
    });
}

function upGML(label_no, posno, id, btn){
    if(btn != 'btn-light'){

    }else{
        $.ajax({
            url: 'http://127.0.0.1:8081/?id='+id+'&posno='+posno+'&label_id='+label_no+'&newlabel=no&ds='+getQueryVariable('ds'),
            type: 'GET',
            success: function(res){
                if(res == 'success'){
                    window.location.href = window.location.href;
                }
            }
        });
        setTimeout(function(){ console.log("here"); location.reload(true) }, 3000);
    }
}

function getQueryVariable(variable) {
    var query = window.location.search.substring(1);
    var vars = query.split('&');
    for (var i = 0; i < vars.length; i++) {
        var pair = vars[i].split('=');
        if (decodeURIComponent(pair[0]) == variable) {
            return decodeURIComponent(pair[1]);
        }
    }
    console.log('Query variable %s not found', variable);
}

function toggle_ds(){
    if(getQueryVariable("ds") == 1){
        $(location).attr('href', 'http://localhost:8000/?ds=2');
    }else{
        $(location).attr('href', 'http://localhost:8000/?ds=1');
    }
}