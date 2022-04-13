package require pbctools

proc center-no-wrap { refSelText zofftext wrapSelText zoffset} {
    set total [atomselect top all]
    set ref [atomselect top $refSelText]
    set oref [atomselect top $zofftext]
    set nf [molinfo top get numframes]
    puts [format "%i frames\n" $nf]
    # Center the reference system around (0, 0, 0)
    for {set f 0} {$f < $nf} {incr f} {
        $ref frame $f
        $oref frame $f
        $total frame $f
        $total moveby [vecinvert [measure center $ref weight mass]]
        set com1 [ lindex [ vecinvert [measure center $oref weight mass]] 2 ]
        set zoff1 [ expr {$zoffset + $com1} ]
        $total move [transoffset "0 0 $zoff1" ] 
    }
    puts "Reference is centered and moved to z=$zoffset. Wrapping \"$wrapSelText\"..."
#    pbc wrap -sel $wrapSelText -first first -last last -center origin -compound residue
    $ref delete
    $oref delete
    $total delete
} 

#proc center-mprot {} {
 center-no-wrap "protein" "protein" "all" 0
#}